import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import scanpy as sc
import os
import shutil
from typing import Dict, Tuple, List, Any
# Import from your pretrain module
from ..pretrain import TCRVocab, FoundationModel

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


class SingleCellClassificationDataset(Dataset):
    """
    Dataset for classification tasks (Disease or Tissue Metatype).
    Compatible with pretrain model's TCRVocab.
    """
    def __init__(self, adata, tcr_vocab: TCRVocab, label_encoder: Dict[str, int], 
                 label_column: str, max_length: int = 30):
        valid_idx = ~adata.obs[label_column].isna()
        self.adata = adata[valid_idx].copy()
        
        if hasattr(self.adata.X, 'toarray'):
            self.gene_expr = self.adata.X.toarray()
        else:
            self.gene_expr = self.adata.X
        self.gene_expr = torch.FloatTensor(self.gene_expr)
        
        self.max_length = max_length
        self.tcr_vocab = tcr_vocab
        
        self.labels = [label_encoder[label] for label in self.adata.obs[label_column]]
        
    def __len__(self) -> int:
        return self.adata.n_obs
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        genes = self.gene_expr[idx]
        
        # Use the same encoding as in pretrain (with add_special_tokens=True)
        cdr3a = self.tcr_vocab.encode(
            self.adata.obs['CDR3a'].iloc[idx], 
            max_length=self.max_length,
            add_special_tokens=True
        )
        cdr3b = self.tcr_vocab.encode(
            self.adata.obs['CDR3b'].iloc[idx], 
            max_length=self.max_length,
            add_special_tokens=True
        )
        
        label = self.labels[idx]
        return {
            'genes': genes,
            'cdr3a': torch.LongTensor(cdr3a),
            'cdr3b': torch.LongTensor(cdr3b),
            'label': torch.tensor(label, dtype=torch.long)
        }

class RepresentationDataset(Dataset):
    """
    Dataset for representation tasks.
    Compatible with pretrain model's TCRVocab.
    """
    def __init__(self, adata, tcr_vocab: TCRVocab, max_length: int = 30):
        self.adata = adata.copy()
        
        if hasattr(self.adata.X, 'toarray'):
            self.gene_expr = self.adata.X.toarray()
        else:
            self.gene_expr = self.adata.X
        self.gene_expr = torch.FloatTensor(self.gene_expr)
        
        self.max_length = max_length
        self.tcr_vocab = tcr_vocab
        
    def __len__(self) -> int:
        return self.adata.n_obs
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        genes = self.gene_expr[idx]
        
        # Use the same encoding as in pretrain (with add_special_tokens=True)
        cdr3a = self.tcr_vocab.encode(
            self.adata.obs['CDR3a'].iloc[idx], 
            max_length=self.max_length,
            add_special_tokens=True
        )
        cdr3b = self.tcr_vocab.encode(
            self.adata.obs['CDR3b'].iloc[idx], 
            max_length=self.max_length,
            add_special_tokens=True
        )
        
        return {
            'genes': genes,
            'cdr3a': torch.LongTensor(cdr3a),
            'cdr3b': torch.LongTensor(cdr3b)
        }

class MetatypeClassifier(nn.Module):
    """
    Unified classifier for both Disease and Tissue classification tasks.
    Compatible with the new foundation model.
    """
    def __init__(self, foundation_model: nn.Module, num_classes: int, 
                 mode: str = 'rna_tcr', task_name: str = 'metatype'):
        super().__init__()
        self.foundation = foundation_model
        self.mode = mode
        self.task_name = task_name
        
        # Define classifier input dimensions based on modality
        # Updated to match your foundation model's output dimensions
        if mode == 'rna_tcr':
            self.classifier = nn.Linear(256, num_classes)  # foundation_output_dim
        elif mode == 'rna_only':
            self.classifier = nn.Linear(128, num_classes)  # tcr_output_dim
        elif mode == 'tcr_only':
            self.classifier = nn.Linear(256, num_classes)  # tcr_output_dim * 2
        elif mode == 'tcra_only':
            self.classifier = nn.Linear(128, num_classes)  # tcr_output_dim
        elif mode == 'tcrb_only':
            self.classifier = nn.Linear(128, num_classes)  # tcr_output_dim
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def forward(self, genes: torch.Tensor, cdr3a: torch.Tensor, 
                cdr3b: torch.Tensor) -> torch.Tensor:
        # Create dummy/zero tensors for unused modalities
        batch_size = genes.size(0)
        device = genes.device
        
        if self.mode == 'rna_only':
            # Use zeros for TCR inputs when using RNA only
            dummy_cdr3a = torch.zeros_like(cdr3a)
            dummy_cdr3b = torch.zeros_like(cdr3b)
            gene_feat, _, _, _ = self.foundation(genes, dummy_cdr3a, dummy_cdr3b)
            x = gene_feat
        
        elif self.mode == 'tcra_only':
            # Use zeros for gene and TCR-B inputs when using TCR-A only
            dummy_genes = torch.zeros_like(genes)
            dummy_cdr3b = torch.zeros_like(cdr3b)
            _, tcr_a_feat, _, _ = self.foundation(dummy_genes, cdr3a, dummy_cdr3b)
            x = tcr_a_feat
        
        elif self.mode == 'tcrb_only':
            # Use zeros for gene and TCR-A inputs when using TCR-B only
            dummy_genes = torch.zeros_like(genes)
            dummy_cdr3a = torch.zeros_like(cdr3a)
            _, _, tcr_b_feat, _ = self.foundation(dummy_genes, dummy_cdr3a, cdr3b)
            x = tcr_b_feat
        
        elif self.mode == 'tcr_only':
            # Use zeros for gene inputs when using TCR only
            dummy_genes = torch.zeros_like(genes)
            _, tcr_a_feat, tcr_b_feat, _ = self.foundation(dummy_genes, cdr3a, cdr3b)
            x = torch.cat((tcr_a_feat, tcr_b_feat), dim=1)
        
        elif self.mode == 'rna_tcr':
            # Use all modalities for RNA+TCR mode
            gene_feat, tcr_a_feat, tcr_b_feat, fused_feat = self.foundation(genes, cdr3a, cdr3b)
            x = fused_feat
        
        return self.classifier(x)

def train_epoch(model: nn.Module, dataloader: DataLoader, 
                optimizer: optim.Optimizer, criterion: nn.Module, 
                device: torch.device) -> float:
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        genes = batch['genes'].to(device)
        cdr3a = batch['cdr3a'].to(device)
        cdr3b = batch['cdr3b'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        logits = model(genes, cdr3a, cdr3b)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * genes.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate(model: nn.Module, dataloader: DataLoader, 
             criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    """
    Evaluate the model on a dataset.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            genes = batch['genes'].to(device)
            cdr3a = batch['cdr3a'].to(device)
            cdr3b = batch['cdr3b'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(genes, cdr3a, cdr3b)
            loss = criterion(logits, labels)
            total_loss += loss.item() * genes.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total if total > 0 else 0
    return total_loss / len(dataloader.dataset), accuracy

def compute_metrics(model: nn.Module, dataloader: DataLoader, 
                   device: torch.device) -> Tuple[float, float, float]:
    """
    Compute various metrics for model evaluation.
    """
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            genes = batch['genes'].to(device)
            cdr3a = batch['cdr3a'].to(device)
            cdr3b = batch['cdr3b'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(genes, cdr3a, cdr3b)
            preds = torch.argmax(logits, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    return acc, f1_macro, f1_weighted

def extract_embeddings_and_predictions(adata, checkpoint_path, model_paths, 
                                     label_column, label_encoder, batch_size=1024, 
                                     max_length=30):
    """
    Extract embeddings and predictions from fine-tuned models and store in adata.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load foundation model and vocabulary
    foundation_model, tcr_vocab = load_foundation_model(checkpoint_path)
    
    # Create dataset
    dataset = RepresentationDataset(adata, tcr_vocab, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Reverse label encoder for predictions
    idx_to_label = {idx: label for label, idx in label_encoder.items()}
    task_name = label_column.lower().replace(' ', '_')
    
    # For each modality, load the fine-tuned model and extract embeddings + predictions
    for mode, model_path in model_paths.items():
        print(f"\nExtracting embeddings and predictions for {mode} modality...")
        
        # Load the saved state dict first to get the correct number of classes
        state_dict = torch.load(model_path, map_location=device)
        num_classes = state_dict['classifier.weight'].size(0)
        
        # Initialize model with correct number of classes
        model = MetatypeClassifier(foundation_model, num_classes, mode=mode).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        
        # Storage for embeddings and predictions
        all_predictions = []
        all_embeddings = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Processing {mode}"):
                genes = batch['genes'].to(device)
                cdr3a = batch['cdr3a'].to(device)
                cdr3b = batch['cdr3b'].to(device)
                
                # Get predictions
                logits = model(genes, cdr3a, cdr3b)
                preds = torch.argmax(logits, dim=1)
                all_predictions.extend(preds.cpu().numpy())
                
                # Get embeddings based on mode
                gene_feat, tcr_a_feat, tcr_b_feat, fused_feat = model.foundation(genes, cdr3a, cdr3b)
                
                if mode == 'rna_tcr':
                    all_embeddings.append(fused_feat.cpu().numpy())
                elif mode == 'rna_only':
                    all_embeddings.append(gene_feat.cpu().numpy())
                elif mode == 'tcr_only':
                    combined_feat = torch.cat((tcr_a_feat, tcr_b_feat), dim=1)
                    all_embeddings.append(combined_feat.cpu().numpy())
                elif mode == 'tcra_only':
                    all_embeddings.append(tcr_a_feat.cpu().numpy())
                elif mode == 'tcrb_only':
                    all_embeddings.append(tcr_b_feat.cpu().numpy())
        
        # Store embeddings
        embeddings = np.concatenate(all_embeddings)
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(adata.n_obs, -1)
        adata.obsm[f'X_{task_name}_{mode}_emb'] = embeddings
        
        # Store predictions as labels
        pred_labels = [idx_to_label[pred] for pred in all_predictions]
        adata.obs[f'{task_name}_{mode}_pred'] = pred_labels
        
        print(f"Stored embedding in adata.obsm['X_{task_name}_{mode}_emb'] with shape {embeddings.shape}")
        print(f"Stored predictions in adata.obs['{task_name}_{mode}_pred']")

def train_classifier(adata, label_column: str, checkpoint_path: str, 
                    num_epochs: int = 50, batch_size: int = 64, 
                    modalities = ['rna_only', 'tcr_only', 'tcra_only', 'tcrb_only', 'rna_tcr'], embeddings = True, 
                    custom_splits = None,
                    return_training_history: bool = False) -> Tuple[Dict[str, Any], Any]:
    """
    Main function to train a classifier and return results + updated adata.
    """
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Filter out NA labels
    adata_filtered = adata[~adata.obs[label_column].isna()].copy()
    
    # Build label encoder
    unique_labels = sorted(adata_filtered.obs[label_column].unique().tolist())
    label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
    num_classes = len(label_encoder)
    print(f"Number of classes: {num_classes}, Classes: {unique_labels}")
    
    # Load checkpoint to determine max_length
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    max_length = checkpoint.get('max_length', 30)
    
    # Load foundation model and vocabulary once
    foundation_model, tcr_vocab = load_foundation_model(checkpoint_path)
    
    # Create dataset with proper TCRVocab object
    dataset = SingleCellClassificationDataset(adata_filtered, tcr_vocab, label_encoder, label_column, max_length=max_length)
    
    # Split data
    if custom_splits is None:
        # Use default splitting strategy
        indices = np.arange(len(dataset))
        train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=0)
        val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=0)
    else:
        # Use provided custom splits
        train_idx, val_idx, test_idx = custom_splits
        train_idx = np.array(train_idx) if not isinstance(train_idx, np.ndarray) else train_idx
        val_idx = np.array(val_idx) if not isinstance(val_idx, np.ndarray) else val_idx
        test_idx = np.array(test_idx) if not isinstance(test_idx, np.ndarray) else test_idx
        print(f"Using custom splits: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # Create data subsets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    
    # Train for each modality
    results = {}
    model_paths = {}
    task_name = label_column
    
    for mode in modalities:
        print(f"\n==================== Training {task_name} classifier: {mode} ====================")
        
        # Create fresh foundation model for each modality
        foundation_copy, _ = load_foundation_model(checkpoint_path)
        foundation_copy.train()
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=10*batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=10*batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        # Initialize model and training components
        model = MetatypeClassifier(foundation_copy, num_classes, mode=mode, task_name=task_name).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Create directory for saving models
        os.makedirs("fine_tune_models", exist_ok=True)
        
        # Training loop
        best_val_acc = 0.0
        best_model_path = f"fine_tune_models/best_{task_name.lower().replace(' ', '_')}_classifier_{mode}.pt"
        model_paths[mode] = best_model_path
        
        # Initialize history tracking if requested
        training_history = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": []
        } if return_training_history else None
        
        for epoch in range(num_epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            print(f"Mode {mode} Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f} | Val Acc = {val_acc:.4f}")
            
            # Record history if requested
            if return_training_history:
                training_history["train_loss"].append(train_loss)
                training_history["val_loss"].append(val_loss)
                training_history["val_acc"].append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                print(f"--> Best model saved with Val Acc = {best_val_acc:.4f}")
        
        # Load best model and compute metrics
        model.load_state_dict(torch.load(best_model_path))
        train_metrics = compute_metrics(model, train_loader, device)
        val_metrics = compute_metrics(model, val_loader, device)
        test_metrics = compute_metrics(model, test_loader, device)
        results[mode] = {"train": train_metrics, "val": val_metrics, "test": test_metrics}
        
        # Add history to results if requested
        if return_training_history:
            results[mode]["history"] = training_history
    
    # Extract embeddings and predictions for all cells (including those with NA labels)
    print(f"\n=== Extracting embeddings and predictions for {task_name} ===")
    adata_new = adata.copy()  # Work with the full dataset
    if embeddings:
    # For cells with NA labels, we still want embeddings but predictions will be for the non-NA subset
        extract_embeddings_and_predictions(
            adata_new, checkpoint_path, model_paths, label_column, 
            label_encoder, batch_size=1024, max_length=max_length
        )
    
    return results, adata_new

def extract_and_store_embeddings(adata, checkpoint_path, model_paths, batch_size=1024, max_length=30):
    """
    Extract embeddings from fine-tuned model and store them in adata.obsm.
    Uses RepresentationDataset which doesn't require labels.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load foundation model and vocabulary
    foundation_model, tcr_vocab = load_foundation_model(checkpoint_path)
    
    # Create dataset - use RepresentationDataset which doesn't need labels
    dataset = RepresentationDataset(adata, tcr_vocab, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # For each modality, load the fine-tuned model and extract embeddings
    for mode, model_path in model_paths.items():
        print(f"\nExtracting embeddings for {mode} modality...")
        
        # Load the saved state dict first to get the correct number of classes
        state_dict = torch.load(model_path, map_location=device)
        # Get number of classes from the classifier weight shape
        num_classes = state_dict['classifier.weight'].size(0)
        
        # Initialize model with correct number of classes
        model = MetatypeClassifier(foundation_model, num_classes, mode=mode).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        
        # Storage for embeddings based on model type
        gene_embs = []
        tcr_a_embs = []
        tcr_b_embs = []
        fused_embs = []
        
        # Extract only needed embeddings based on model type
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Processing {mode}"):
                genes = batch['genes'].to(device)
                cdr3a = batch['cdr3a'].to(device)
                cdr3b = batch['cdr3b'].to(device)
                
                # Get all embeddings from foundation model
                gene_feat, tcr_a_feat, tcr_b_feat, fused_feat = model.foundation(genes, cdr3a, cdr3b)
                
                # Store the relevant embeddings based on modality
                if mode == 'rna_tcr':
                    fused_embs.append(fused_feat.cpu().numpy())
                    
                elif mode == 'rna_only':
                    gene_embs.append(gene_feat.cpu().numpy())
                    
                elif mode == 'tcr_only':
                    tcr_a_embs.append(tcr_a_feat.cpu().numpy())
                    tcr_b_embs.append(tcr_b_feat.cpu().numpy())
                    
                elif mode == 'tcra_only':
                    tcr_a_embs.append(tcr_a_feat.cpu().numpy())
                
                elif mode == 'tcrb_only':
                    tcr_b_embs.append(tcr_b_feat.cpu().numpy())
        
        # Process and store the collected embeddings
        if mode == 'rna_tcr':
            fused_embs = np.concatenate(fused_embs)
            print(f"fused_embs shape: {fused_embs.shape}")
            print(f"adata shape: {adata.shape}")
            if len(fused_embs.shape) == 1:
                fused_embs = fused_embs.reshape(adata.n_obs, -1)
            adata.obsm[f'X_fused_emb'] = fused_embs
            print(f"Stored fused embedding in adata.obsm['X_fused_emb']")
            
        elif mode == 'rna_only':
            gene_embs = np.concatenate(gene_embs)
            if len(gene_embs.shape) == 1:
                gene_embs = gene_embs.reshape(adata.n_obs, -1)
            adata.obsm[f'X_gene_emb'] = gene_embs
            print(f"Stored gene embedding in adata.obsm['X_gene_emb']")
            
        elif mode == 'tcr_only':
            tcr_a_embs = np.concatenate(tcr_a_embs)
            tcr_b_embs = np.concatenate(tcr_b_embs)
            if len(tcr_a_embs.shape) == 1:
                tcr_a_embs = tcr_a_embs.reshape(adata.n_obs, -1)
                tcr_b_embs = tcr_b_embs.reshape(adata.n_obs, -1)
            adata.obsm[f'X_tcr_a_emb'] = tcr_a_embs
            adata.obsm[f'X_tcr_b_emb'] = tcr_b_embs
            print(f"Stored TCR embeddings in adata.obsm['X_tcr_a_emb'] and ['X_tcr_b_emb']")
            
        elif mode == 'tcra_only':
            tcr_a_embs = np.concatenate(tcr_a_embs)
            if len(tcr_a_embs.shape) == 1:
                tcr_a_embs = tcr_a_embs.reshape(adata.n_obs, -1)
            adata.obsm[f'X_tcra_emb'] = tcr_a_embs
            print(f"Stored TCR alpha embedding in adata.obsm['X_tcra_emb']")
            
        elif mode == 'tcrb_only':
            tcr_b_embs = np.concatenate(tcr_b_embs)
            if len(tcr_b_embs.shape) == 1:
                tcr_b_embs = tcr_b_embs.reshape(adata.n_obs, -1)
            adata.obsm[f'X_tcrb_emb'] = tcr_b_embs
            print(f"Stored TCR beta embedding in adata.obsm['X_tcrb_emb']")
            
    return adata