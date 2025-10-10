import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Any

# Import existing functions from cla_funcs
from .classification import load_foundation_model
# from pretrain import TCRVocab

class BindingCountsRegressor(nn.Module):
    """
    Regressor for predicting binding counts.
    Adapted to work with the foundation model.
    """
    def __init__(self, foundation_model: nn.Module, output_dim: int = 8, 
                 mode: str = 'rna_tcr', task_name: str = 'binding_counts'):
        super().__init__()
        self.foundation = foundation_model
        self.mode = mode
        self.task_name = task_name
        
        # Define regressor input dimensions based on modality
        if mode == 'rna_tcr':
            self.regressor = nn.Sequential(
                nn.LayerNorm(256),  # Input normalization
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, output_dim)
            )
        elif mode == 'rna_only':
            self.regressor = nn.Sequential(
                nn.LayerNorm(128),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, output_dim)
            )
        elif mode == 'tcr_only':
            self.regressor = nn.Sequential(
                nn.LayerNorm(256),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, output_dim)
            )
        elif mode == 'tcra_only':
            self.regressor = nn.Sequential(
                nn.LayerNorm(128),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, output_dim)
            )
        elif mode == 'tcrb_only':
            self.regressor = nn.Sequential(
                nn.LayerNorm(128),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, output_dim)
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def forward(self, genes: torch.Tensor, cdr3a: torch.Tensor, 
                cdr3b: torch.Tensor) -> torch.Tensor:
        # Create dummy/zero tensors for unused modalities
        batch_size = genes.size(0)
        device = genes.device
        
        # For modality-specific modes, only pass the relevant data to the foundation model
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
        
        return self.regressor(x)

class BindingCountsDataset(Dataset):
    """
    Dataset for binding counts regression tasks.
    """
    def __init__(self, adata, tcr_vocab):
        self.adata = adata.copy()
        
        if hasattr(self.adata.X, 'toarray'):
            self.gene_expr = self.adata.X.toarray()
        else:
            self.gene_expr = self.adata.X
        self.gene_expr = torch.FloatTensor(self.gene_expr)
        
        self.tcr_vocab = tcr_vocab
        
        self.cdr3a_list = self.adata.obs['CDR3a'].tolist()
        self.cdr3b_list = self.adata.obs['CDR3b'].tolist()
        
        # Get binding counts
        self.binding_counts = torch.FloatTensor(self.adata.obsm['binding_counts'])
        
    def __len__(self) -> int:
        return self.adata.n_obs
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        genes = self.gene_expr[idx]
        cdr3a_encoded = self.tcr_vocab.encode(self.cdr3a_list[idx])
        cdr3b_encoded = self.tcr_vocab.encode(self.cdr3b_list[idx])
        counts = self.binding_counts[idx]
        return {
            'genes': genes,
            'cdr3a': torch.LongTensor(cdr3a_encoded),
            'cdr3b': torch.LongTensor(cdr3b_encoded),
            'binding_counts': counts
        }

def train_epoch_regression(model: nn.Module, dataloader: DataLoader, 
                           optimizer: optim.Optimizer, criterion: nn.Module, 
                           device: torch.device) -> float:
    """
    Train the regression model for one epoch.
    """
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        genes = batch['genes'].to(device)
        cdr3a = batch['cdr3a'].to(device)
        cdr3b = batch['cdr3b'].to(device)
        binding_counts = batch['binding_counts'].to(device)
        
        optimizer.zero_grad()
        predictions = model(genes, cdr3a, cdr3b)
        loss = criterion(predictions, binding_counts)
        # loss = F.smooth_l1_loss(predictions, binding_counts)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * genes.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate_regression(model: nn.Module, dataloader: DataLoader, 
                        criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    """
    Evaluate the regression model on a dataset.
    Returns loss and R² score.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            genes = batch['genes'].to(device)
            cdr3a = batch['cdr3a'].to(device)
            cdr3b = batch['cdr3b'].to(device)
            binding_counts = batch['binding_counts'].to(device)
            
            predictions = model(genes, cdr3a, cdr3b)
            loss = criterion(predictions, binding_counts)
            total_loss += loss.item() * genes.size(0)
            
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(binding_counts.cpu().numpy())
    
    # Calculate R² score
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Calculate R² for each binding count separately
    r2_scores = []
    for i in range(all_preds.shape[1]):
        # Skip columns with all zeros in targets
        if np.all(all_targets[:, i] == 0):
            r2_scores.append(0.0)
            continue
            
        # Calculate R²
        target_mean = np.mean(all_targets[:, i])
        ss_tot = np.sum((all_targets[:, i] - target_mean) ** 2)
        ss_res = np.sum((all_targets[:, i] - all_preds[:, i]) ** 2)
        
        if ss_tot == 0:  # Avoid division by zero
            r2_scores.append(0.0)
        else:
            r2_scores.append(1 - (ss_res / ss_tot))
    
    # Average R² across all binding counts
    avg_r2 = np.mean(r2_scores)
    return total_loss / len(dataloader.dataset), avg_r2

def compute_msle(y_true, y_pred):
    """
    Compute Mean Squared Logarithmic Error safely handling zeros, negative values,
    and potential numerical issues.
    
    MSLE = mean((log(max(y_true + 1, 1e-8)) - log(max(y_pred + 1, 1e-8)))²)
    """
    # Ensure no negative values by clipping
    y_true_safe = np.clip(y_true, 0.0, None)
    y_pred_safe = np.clip(y_pred, 0.0, None)
    
    # Add 1 to handle zeros (log1p = log(1+x))
    log_true = np.log1p(y_true_safe)
    log_pred = np.log1p(y_pred_safe)
    
    # Replace any potential NaN or inf values
    log_true = np.nan_to_num(log_true, nan=0.0, posinf=0.0, neginf=0.0)
    log_pred = np.nan_to_num(log_pred, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Compute mean squared difference
    msle = np.mean((log_true - log_pred) ** 2)
    
    return msle
def compute_regression_metrics(model: nn.Module, dataloader: DataLoader, 
                              device: torch.device) -> Dict[str, float]:
    """
    Compute various metrics for regression model evaluation including MSLE.
    """
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in dataloader:
            genes = batch['genes'].to(device)
            cdr3a = batch['cdr3a'].to(device)
            cdr3b = batch['cdr3b'].to(device)
            binding_counts = batch['binding_counts'].to(device)
            
            predictions = model(genes, cdr3a, cdr3b)
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(binding_counts.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Calculate metrics for each binding count
    metrics = {
        'r2_scores': [],
        'mse': [],
        'mae': [],
        'msle': []
    }
    
    for i in range(all_preds.shape[1]):
        # Calculate R²
        target_mean = np.mean(all_targets[:, i])
        ss_tot = np.sum((all_targets[:, i] - target_mean) ** 2)
        ss_res = np.sum((all_targets[:, i] - all_preds[:, i]) ** 2)
        
        if ss_tot == 0:  # Avoid division by zero
            r2 = 0.0
        else:
            r2 = 1 - (ss_res / ss_tot)
        
        # Calculate MSE and MAE
        mse = np.mean((all_targets[:, i] - all_preds[:, i]) ** 2)
        mae = np.mean(np.abs(all_targets[:, i] - all_preds[:, i]))
        
        # Calculate MSLE
        msle = compute_msle(all_targets[:, i], all_preds[:, i])
        
        metrics['r2_scores'].append(r2)
        metrics['mse'].append(mse)
        metrics['mae'].append(mae)
        metrics['msle'].append(msle)
    
    # Average metrics
    avg_metrics = {
        'avg_r2': np.mean(metrics['r2_scores']),
        'avg_mse': np.mean(metrics['mse']),
        'avg_mae': np.mean(metrics['mae']),
        'avg_msle': np.mean(metrics['msle']),
        'per_output_r2': metrics['r2_scores'],
        'per_output_mse': metrics['mse'],
        'per_output_mae': metrics['mae'],
        'per_output_msle': metrics['msle']
    }
    
    return avg_metrics

def train_binding_counts_regressor(adata, checkpoint_path: str, 
                                  num_epochs: int = 50, batch_size: int = 64,
                                  modalities = ['rna_only', 'tcr_only', 'tcra_only', 'tcrb_only', 'rna_tcr'],
                                  custom_splits = None,
                                  return_training_history: bool = False) -> Dict[str, Any]:
    """
    Main function to train a regressor for binding counts prediction.
    
    Args:
        adata: AnnData object containing the data with binding_counts in obsm
        checkpoint_path: Path to the foundation model checkpoint
        num_epochs: Number of epochs to train
        batch_size: Batch size for training
        modalities: List of modalities to train on
        custom_splits: Optional tuple of (train_idx, val_idx, test_idx) for custom data splits
        return_training_history: If True, return training losses and metrics for each epoch
    
    Returns:
        dict: Results containing metrics for each modality and optionally training history
    """
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure binding_counts is available in obsm
    if 'binding_counts' not in adata.obsm:
        raise ValueError("binding_counts not found in adata.obsm")
    
    # Get output dimension from binding_counts
    output_dim = adata.obsm['binding_counts'].shape[1]
    print(f"Output dimension: {output_dim}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    
    # Create initial dataset with dummy vocab
    dummy_vocab = {}
    dataset = BindingCountsDataset(adata, dummy_vocab)
    
    # Split data
    if custom_splits is None:
        # Use default splitting strategy
        indices = np.arange(len(dataset))
        train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=0)
        val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=0)
    else:
        # Use provided custom splits - convert to standard numpy arrays if needed
        train_idx, val_idx, test_idx = custom_splits
        train_idx = np.array(train_idx) if not isinstance(train_idx, np.ndarray) else train_idx
        val_idx = np.array(val_idx) if not isinstance(val_idx, np.ndarray) else val_idx
        test_idx = np.array(test_idx) if not isinstance(test_idx, np.ndarray) else test_idx
        print(f"Using custom splits: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    # Train for each modality
    results = {}
    task_name = 'binding_counts'
    
    for mode in modalities:
        print(f"\n==================== Training {task_name} regressor: {mode} ====================")
        
        # Load foundation model and vocabulary
        foundation_model, vocab = load_foundation_model(checkpoint_path)
        foundation_model.train()
        
        # Create dataset with correct vocabulary
        dataset = BindingCountsDataset(adata, vocab)
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        test_dataset = Subset(dataset, test_idx)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        # Initialize model and training components
        model = BindingCountsRegressor(foundation_model, output_dim=output_dim, mode=mode, task_name=task_name).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()  # Mean Squared Error for regression
        
        # Training loop
        best_val_r2 = -float('inf')  # R² can be negative, so start with negative infinity
        best_model_path = f"fine_tune_models/best_{task_name}_regressor_{mode}.pt"
        
        # Initialize history tracking if requested
        training_history = {
            "train_loss": [],
            "val_loss": [],
            "val_r2": []
        } if return_training_history else None
        
        for epoch in range(num_epochs):
            train_loss = train_epoch_regression(model, train_loader, optimizer, criterion, device)
            val_loss, val_r2 = evaluate_regression(model, val_loader, criterion, device)
            print(f"Mode {mode} Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f} | Val R² = {val_r2:.4f}")
            
            # Record history if requested
            if return_training_history:
                training_history["train_loss"].append(train_loss)
                training_history["val_loss"].append(val_loss)
                training_history["val_r2"].append(val_r2)
            
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                torch.save(model.state_dict(), best_model_path)
                print(f"--> Best model saved with Val R² = {best_val_r2:.4f}")
        
        # Load best model and compute metrics
        model.load_state_dict(torch.load(best_model_path))
        train_metrics = compute_regression_metrics(model, train_loader, device)
        val_metrics = compute_regression_metrics(model, val_loader, device)
        test_metrics = compute_regression_metrics(model, test_loader, device)
        
        results[mode] = {"train": train_metrics, "val": val_metrics, "test": test_metrics}
        
        # Add history to results if requested
        if return_training_history:
            results[mode]["history"] = training_history
    
    return results

def build_regression_results_dataframe(results):
    """
    Build a DataFrame from regression results for easier visualization.
    """
    data = []
    for mode in results:
        for split in ['train', 'val', 'test']:
            metrics = results[mode][split]
            data.append({
                'Mode': mode,
                'Split': split,
                'R²': metrics['avg_r2'],
                'MSE': metrics['avg_mse'],
                'MAE': metrics['avg_mae'],
                'MSLE': metrics['avg_msle']
            })
    
    return pd.DataFrame(data)

def plot_regression_metrics_charts(df, output_dir, task_name):
    """
    Create visualization charts for regression metrics including MSLE
    using specified model names and colors.
    """
    # Define color mapping with proper model names
    colors = {
        'TCRfoundation': '#FF8A80',
        'TCR α+β': '#91D1C2',  
        'TCR α only': '#B5A8FF',  
        'TCR β only': '#81C784',
        'mvTCR': '#FFD54F',
        'RNA only': '#4DBBD5',
    }
    
    # Create mapping between internal mode names and display names
    mode_to_display = {
        'rna_tcr': 'TCRfoundation',
        'tcr_only': 'TCR α+β',
        'tcra_only': 'TCR α only',
        'tcrb_only': 'TCR β only',
        'rna_only': 'RNA only'
    }
    
    # Create a copy of the dataframe with display names
    df_display = df.copy()
    df_display['Display_Mode'] = df_display['Mode'].map(mode_to_display)
    
    # Create color palette for the display names
    palette = {name: color for name, color in colors.items() if name in df_display['Display_Mode'].unique()}
    
    plt.figure(figsize=(16, 10))
    
    # R² Score plot
    plt.subplot(2, 2, 1)
    sns.barplot(x='Display_Mode', y='R²', hue='Split', data=df_display, 
                hue_order=['train', 'val', 'test'], palette=palette)
    plt.title(f'{task_name} - R² Score by Mode and Split')
    plt.xticks(rotation=45)
    plt.ylim(0, 1.0)  # R² ideally ranges from 0 to 1
    
    # MSE plot
    plt.subplot(2, 2, 2)
    sns.barplot(x='Display_Mode', y='MSE', hue='Split', data=df_display, 
                hue_order=['train', 'val', 'test'], palette=palette)
    plt.title(f'{task_name} - Mean Squared Error by Mode and Split')
    plt.xticks(rotation=45)
    
    # MAE plot
    plt.subplot(2, 2, 3)
    sns.barplot(x='Display_Mode', y='MAE', hue='Split', data=df_display, 
                hue_order=['train', 'val', 'test'], palette=palette)
    plt.title(f'{task_name} - Mean Absolute Error by Mode and Split')
    plt.xticks(rotation=45)
    
    # MSLE plot
    plt.subplot(2, 2, 4)
    if df_display['MSLE'].notna().any():  # Check if there are any valid MSLE values
        sns.barplot(x='Display_Mode', y='MSLE', hue='Split', data=df_display, 
                    hue_order=['train', 'val', 'test'], palette=palette)
        plt.title(f'{task_name} - Mean Squared Log Error by Mode and Split')
    else:
        plt.text(0.5, 0.5, 'MSLE values are all NaN\nConsider using a different metric', 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=12)
        plt.title(f'{task_name} - MSLE (Not Available)')
    plt.xticks(rotation=45)
    
    # Add a legend for mode colors to the first subplot
    ax1 = plt.subplot(2, 2, 1)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, title="Split", loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{task_name}_regression_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()