import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import scanpy as sc
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error
from typing import Dict, Tuple, List, Any
from ..pretrain import TCRVocab, FoundationModel

class TCR2GeneRegressor(nn.Module):
    """
    A model that takes TCR sequences and predicts gene expression.
    Uses the pretrained foundation model's TCR encoders and gene decoder.
    Fine-tunes both the TCR encoder and gene decoder.
    """
    def __init__(self, foundation_model, gene_decoder, gene_dim, modality="both", freeze_tcr_encoder=False):
        super().__init__()
        self.modality = modality.lower()
        
        # Extract only the needed components from foundation model
        self.tcr_encoder_a = foundation_model.tcr_encoder_a
        self.tcr_encoder_b = foundation_model.tcr_encoder_b
        self.gene_decoder = gene_decoder
        
        # Option to freeze TCR encoder parameters if needed
        if freeze_tcr_encoder:
            for param in self.tcr_encoder_a.parameters():
                param.requires_grad = False
            for param in self.tcr_encoder_b.parameters():
                param.requires_grad = False
            
        # Get the output dimensions from the TCR encoders
        tcr_output_dim = self.tcr_encoder_a.fc.out_features  # Should be 128
        
        # Define modality-specific projection layers
        if self.modality == "tcr_only":
            # Use both TCR chains
            input_dim = tcr_output_dim * 2
        elif self.modality == "tcra_only":
            # Use only alpha chain
            input_dim = tcr_output_dim
        elif self.modality == "tcrb_only":
            # Use only beta chain
            input_dim = tcr_output_dim
        else:
            raise ValueError(f"Unsupported modality: {modality}")
            
        # Create projections from TCR representations to foundation latent space (256 dim)
        self.tcr2latent_projection = nn.Sequential(
            nn.Linear(input_dim, 256),  # Project to latent space expected by gene decoder
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, cdr3a, cdr3b):
        # Get TCR encodings directly from the TCR encoders
        tcr_a_feat = self.tcr_encoder_a(cdr3a)
        tcr_b_feat = self.tcr_encoder_b(cdr3b)
        
        # Select features based on modality
        if self.modality == "tcr_only":
            embeddings = torch.cat([tcr_a_feat, tcr_b_feat], dim=1)
        elif self.modality == "tcra_only":
            embeddings = tcr_a_feat
        elif self.modality == "tcrb_only":
            embeddings = tcr_b_feat
        
        # Project to latent space expected by gene decoder
        latent = self.tcr2latent_projection(embeddings)
        
        # Decode to gene expression using pretrained decoder
        gene_expression = self.gene_decoder(latent)
        
        return gene_expression

class TCR2GeneDataset(Dataset):
    """Dataset for TCR2Gene regression task."""
    def __init__(self, adata, tcr_vocab, max_length=30, add_special_tokens=True):
        self.adata = adata
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        
        # Extract gene expression
        self.gene_expr = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        self.gene_expr = torch.FloatTensor(self.gene_expr)
        
        # Use the provided TCR vocabulary
        self.tcr_vocab = tcr_vocab
        
    def __len__(self):
        return self.adata.n_obs
    
    def __getitem__(self, idx):
        genes = self.gene_expr[idx]
        cdr3a = self.tcr_vocab.encode(
            self.adata.obs['CDR3a'].iloc[idx], 
            max_length=self.max_length,
            add_special_tokens=self.add_special_tokens
        )
        cdr3b = self.tcr_vocab.encode(
            self.adata.obs['CDR3b'].iloc[idx], 
            max_length=self.max_length,
            add_special_tokens=self.add_special_tokens
        )
        return {
            'genes': genes,
            'cdr3a': torch.LongTensor(cdr3a),
            'cdr3b': torch.LongTensor(cdr3b)
        }

def load_foundation_model(checkpoint_path: str) -> Tuple[nn.Module, TCRVocab]:
    """
    Load a pretrained foundation model and its vocabulary.
    
    Args:
        checkpoint_path: Path to the model checkpoint

    Returns:
        Tuple containing:
            - The loaded foundation model
            - The TCRVocab object (not just dictionary)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model = FoundationModel(
        gene_dim=checkpoint['gene_dim'],
        tcr_vocab_size=checkpoint['tcr_vocab_size'],
        max_length=checkpoint.get('max_length', 30),
        dropout=0.1,
        pad_idx=0
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Reconstruct TCRVocab object from checkpoint
    tcr_vocab = TCRVocab()
    tcr_vocab.vocab = checkpoint['vocab']
    tcr_vocab.inverse_vocab = {v: k for k, v in checkpoint['vocab'].items()}
    # Set special token indices based on the vocab
    tcr_vocab.pad_token_idx = checkpoint['vocab']['<pad>']
    tcr_vocab.unk_token_idx = checkpoint['vocab']['<unk>']
    tcr_vocab.start_token_idx = checkpoint['vocab']['<start>']
    tcr_vocab.end_token_idx = checkpoint['vocab']['<end>']
    
    # Get the max_length from the checkpoint
    max_length = checkpoint.get('max_length', 30)
    print(f"Loaded model with max_length: {max_length}")
    
    return model, tcr_vocab

def load_gene_decoder(checkpoint_path):
    """Load the pretrained gene decoder."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gene_dim = checkpoint['gene_dim']
    
    # Create a wrapper for the gene decoder
    class GeneDecoder(nn.Module):
        def __init__(self, decoder_state_dict):
            super().__init__()
            self.decoder = nn.Sequential(
                nn.Linear(256, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Linear(512, gene_dim),
                nn.GELU()
            )
            self.decoder.load_state_dict(decoder_state_dict)
            
        def forward(self, x):
            return self.decoder(x)
    
    # Create and return the decoder
    decoder = GeneDecoder(checkpoint['gene_decoder_state_dict'])
    return decoder

def train_regressor(adata, checkpoint_path, num_epochs=50, batch_size=1024, 
                    learning_rate=1e-4, modalities=None, val_split=0.1, test_split=0.1,
                    save_splits=True, save_predictions=True):
    """
    Train a TCR to gene expression regressor using the pretrained foundation model.
    
    Args:
        adata: AnnData object with gene expression and TCR sequences
        checkpoint_path: Path to the pretrained foundation model checkpoint
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        modalities: List of modalities to train ("tcr_only", "tcra_only", "tcrb_only")
        val_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
        save_splits: Whether to save train/val/test splits in adata.obs
        save_predictions: Whether to save predictions in adata.obsm
        
    Returns:
        Dictionary of training results for each modality and updated adata
    """
    if modalities is None:
        modalities = ["tcr_only", "tcra_only", "tcrb_only"]
    
    # Load the pretrained foundation model and gene decoder
    foundation_model, tcr_vocab = load_foundation_model(checkpoint_path)
    gene_decoder = load_gene_decoder(checkpoint_path)
    
    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    foundation_model = foundation_model.to(device)
    gene_decoder = gene_decoder.to(device)
    
    # Create dataset
    dataset = TCR2GeneDataset(adata, tcr_vocab)
    
    # Split the dataset
    train_size = int((1 - val_split - test_split) * len(dataset))
    val_size = int(val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    # Use fixed seed for reproducibility
    generator = torch.Generator().manual_seed(0)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    # Record split information in adata if requested
    if save_splits:
        # Create a split column in adata.obs
        adata.obs['split'] = 'train'  # Default all to train
        
        # Update with validation and test indices
        val_indices = val_dataset.indices
        test_indices = test_dataset.indices
        
        adata.obs.iloc[val_indices, adata.obs.columns.get_loc('split')] = 'val'
        adata.obs.iloc[test_indices, adata.obs.columns.get_loc('split')] = 'test'
        
        print(f"Split information saved in adata.obs['split']:")
        print(f"  - Train: {(adata.obs['split'] == 'train').sum()} cells")
        print(f"  - Val: {(adata.obs['split'] == 'val').sum()} cells")
        print(f"  - Test: {(adata.obs['split'] == 'test').sum()} cells")
    
    # Create data loaders
    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    # Track results for each modality
    results = {}
    
    # Train a model for each modality
    for modality in modalities:
        print(f"\n=== Training {modality} regressor ===")
        
        # Create the regressor model with fine-tunable components
        regressor = TCR2GeneRegressor(
            foundation_model=foundation_model,
            gene_decoder=gene_decoder,
            gene_dim=adata.n_vars,
            modality=modality,
            freeze_tcr_encoder=False  # Allow fine-tuning of TCR encoder
        )
        regressor = regressor.to(device)
        
        # Set up optimizer
        optimizer = torch.optim.Adam(regressor.parameters(), lr=learning_rate)
        
        # Set up scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Track best validation loss
        best_val_loss = float('inf')
        modality_results = {"train": {}, "val": {}, "test": {}}
        
        # Training loop
        for epoch in range(num_epochs):
            # Training
            regressor.train()
            train_loss = 0
            train_preds = []
            train_targets = []
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)"):
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass
                gene_pred = regressor(batch['cdr3a'], batch['cdr3b'])
                
                # Compute loss
                loss = F.mse_loss(gene_pred, batch['genes'])
                # loss = F.smooth_l1_loss(gene_pred, batch['genes'])
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Track metrics
                train_loss += loss.item() * batch['genes'].size(0)
                train_preds.append(gene_pred.detach().cpu().numpy())
                train_targets.append(batch['genes'].detach().cpu().numpy())
            
            # Calculate metrics
            train_loss /= len(train_dataset)
            train_preds = np.vstack(train_preds)
            train_targets = np.vstack(train_targets)
            train_mse = mean_squared_error(train_targets, train_preds)
            train_r2 = r2_score(train_targets, train_preds)
            
            # Validation
            regressor.eval()
            val_loss = 0
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)"):
                    # Move batch to device
                    batch = {k: v.to(device) for k, v in batch.items()}
                    
                    # Forward pass
                    gene_pred = regressor(batch['cdr3a'], batch['cdr3b'])
                    
                    # Compute loss
                    loss = F.mse_loss(gene_pred, batch['genes'])
                    
                    # Track metrics
                    val_loss += loss.item() * batch['genes'].size(0)
                    val_preds.append(gene_pred.cpu().numpy())
                    val_targets.append(batch['genes'].cpu().numpy())
            
            # Calculate metrics
            val_loss /= len(val_dataset)
            val_preds = np.vstack(val_preds)
            val_targets = np.vstack(val_targets)
            val_mse = mean_squared_error(val_targets, val_preds)
            val_r2 = r2_score(val_targets, val_preds)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, "
                  f"Train R²: {train_r2:.4f}, "
                  f"Val R²: {val_r2:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': regressor.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    'modality': modality,
                    'gene_dim': adata.n_vars
                }, f"fine_tune_models/best_tcr2gene_regressor_{modality}.pt")
                print(f"Saved best model checkpoint (Val Loss: {val_loss:.6f})")
        
        # Evaluate on test set with best model
        print("\nEvaluating on test set with best model...")
        best_model = TCR2GeneRegressor(
            foundation_model=foundation_model,
            gene_decoder=gene_decoder,
            gene_dim=adata.n_vars,
            modality=modality,
            freeze_tcr_encoder=False
        )
        checkpoint = torch.load(f"fine_tune_models/best_tcr2gene_regressor_{modality}.pt")
        best_model.load_state_dict(checkpoint['model_state_dict'])
        best_model = best_model.to(device)
        best_model.eval()
        
        test_loss = 0
        test_preds = []
        test_targets = []
        test_indices = test_dataset.indices
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass
                gene_pred = best_model(batch['cdr3a'], batch['cdr3b'])
                
                # Compute loss
                loss = F.mse_loss(gene_pred, batch['genes'])
                
                # Track metrics
                test_loss += loss.item() * batch['genes'].size(0)
                test_preds.append(gene_pred.cpu().numpy())
                test_targets.append(batch['genes'].cpu().numpy())
        
        # Calculate metrics
        test_loss /= len(test_dataset)
        test_preds = np.vstack(test_preds)
        test_targets = np.vstack(test_targets)
        test_mse = mean_squared_error(test_targets, test_preds)
        test_r2 = r2_score(test_targets, test_preds)
        
        print(f"Test Loss: {test_loss:.6f}, Test MSE: {test_mse:.6f}, Test R²: {test_r2:.4f}")
        
        # Generate predictions for all data points using the best model
        if save_predictions:
            print(f"\nGenerating predictions for all data with best {modality} model...")
            full_loader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=False,
                num_workers=num_workers,
                persistent_workers=True if num_workers > 0 else False,
                prefetch_factor=2 if num_workers > 0 else None
            )
            all_preds = []
            
            with torch.no_grad():
                for batch in tqdm(full_loader, desc="Predicting"):
                    # Move batch to device
                    batch = {k: v.to(device) for k, v in batch.items()}
                    
                    # Forward pass
                    gene_pred = best_model(batch['cdr3a'], batch['cdr3b'])
                    
                    # Store predictions
                    all_preds.append(gene_pred.cpu().numpy())
            
            # Save predictions to adata
            all_preds = np.vstack(all_preds)
            adata.obsm[f'X_gene_pred_{modality}'] = all_preds
            print(f"Predictions saved to adata.obsm['X_gene_pred_{modality}']")
        
        # Store results
        modality_results["train"]["loss"] = train_loss
        modality_results["train"]["mse"] = train_mse
        modality_results["train"]["r2"] = train_r2
        modality_results["val"]["loss"] = val_loss
        modality_results["val"]["mse"] = val_mse
        modality_results["val"]["r2"] = val_r2
        modality_results["test"]["loss"] = test_loss
        modality_results["test"]["mse"] = test_mse
        modality_results["test"]["r2"] = test_r2
        
        results[modality] = modality_results
    
    return results, adata