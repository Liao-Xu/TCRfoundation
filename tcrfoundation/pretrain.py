import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import scanpy as sc
from tqdm import tqdm
import os
import time
import math
import matplotlib.pyplot as plt

# ----------------------------
# 1. Data Preprocessing
# ----------------------------
class TCRVocab:
    """TCR amino acid sequence vocabulary with extended special tokens."""
    def __init__(self):
        self.amino_acids = set()
        self.special_tokens = ['<pad>', '<unk>', '<start>', '<end>']
        self.pad_token_idx = 0
        self.unk_token_idx = 1
        self.start_token_idx = 2
        self.end_token_idx = 3
        self.vocab = {}
        self.inverse_vocab = {}
    
    def build(self, sequences):
        for seq in sequences:
            self.amino_acids.update(seq)
        self.vocab = {aa: i + len(self.special_tokens) for i, aa in enumerate(sorted(self.amino_acids))}
        for i, tok in enumerate(self.special_tokens):
            self.vocab[tok] = i
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def encode(self, seq, max_length=30, add_special_tokens=False):
        if add_special_tokens:
            content_max_length = max_length - 2
            encoded = [self.vocab.get(aa, self.unk_token_idx) for aa in seq[:content_max_length]]
            encoded = [self.start_token_idx] + encoded + [self.end_token_idx]
        else:
            encoded = [self.vocab.get(aa, self.unk_token_idx) for aa in seq[:max_length]]
        if len(encoded) < max_length:
            encoded += [self.pad_token_idx] * (max_length - len(encoded))
        return encoded[:max_length]
    
    def decode(self, tokens, remove_special=True):
        if remove_special:
            tokens = [t for t in tokens if t >= len(self.special_tokens)]
            return ''.join([self.inverse_vocab.get(t, '') for t in tokens])
        else:
            return ''.join([self.inverse_vocab.get(t, '') for t in tokens])

class SingleCellDataset(Dataset):
    def __init__(self, adata, max_length=30, add_special_tokens=False, keep_sparse=True):
        self.adata = adata
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        
        # Option to keep data sparse until needed
        self.is_sparse = keep_sparse and hasattr(adata.X, 'toarray')
        
        if self.is_sparse:
            # Keep sparse matrix, convert on-the-fly in __getitem__
            self.gene_expr = adata.X
        else:
            # Convert to dense tensor upfront
            self.gene_expr = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
            self.gene_expr = torch.FloatTensor(self.gene_expr)
        
        # Build vocabulary
        self.tcr_vocab = TCRVocab()
        all_sequences = list(adata.obs['CDR3a']) + list(adata.obs['CDR3b'])
        self.tcr_vocab.build(all_sequences)
        
        # Cache the dataset size
        self._len = self.adata.n_obs
        
    def __len__(self):
        """Return the number of samples in the dataset"""
        return self._len
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset"""
        # Convert sparse to dense on-the-fly if needed
        if self.is_sparse:
            # Extract row and convert to dense
            genes = self.gene_expr[idx].toarray().squeeze()
            genes = torch.FloatTensor(genes)
        else:
            genes = self.gene_expr[idx]
            
        # Encode TCR sequences
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

# ----------------------------
# 2. Model Architecture
# ----------------------------
class TCRTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=3, 
                 max_length=30, output_dim=128, dropout=0.1, pad_idx=0):
        super().__init__()
        self.padding_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embedding = nn.Parameter(torch.randn(max_length + 1, d_model) * 0.02)
        self.embedding_dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=dropout,
            activation="gelu",
            batch_first=False,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.max_length = max_length

    def forward(self, x):
        batch_size = x.size(0)
        
        # Create padding mask (True where padding tokens are)
        padding_mask = (x == self.padding_idx)
        
        token_embeddings = self.embedding(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, token_embeddings], dim=1)
        x = x + self.pos_embedding.unsqueeze(0)
        x = self.embedding_dropout(x)
        x = x.transpose(0, 1)
        
        # Add CLS token position to padding mask
        cls_padding = torch.zeros(batch_size, 1, dtype=torch.bool, device=padding_mask.device)
        src_key_padding_mask = torch.cat([cls_padding, padding_mask], dim=1)
        
        # Pass mask to transformer
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        cls_output = x[0]
        out = self.fc(cls_output)
        out = self.layer_norm(out)
        return out

class TCRTransformerDecoder(nn.Module):
    def __init__(self, latent_dim, d_model, max_length, vocab_size, nhead=4, dropout=0.1):
        super().__init__()
        self.max_length = max_length
        self.d_model = d_model
        
        # Project latent to d_model
        self.latent_proj = nn.Linear(latent_dim, d_model)
        
        # Learnable query embeddings for each position
        self.query_embeddings = nn.Parameter(torch.randn(max_length, d_model) * 0.02)
        
        # Single transformer decoder layer
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,  # Smaller FFN
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        
        # Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, z):
        batch_size = z.size(0)
        
        # Memory from latent
        memory = self.latent_proj(z).unsqueeze(1)  # [batch, 1, d_model]
        
        # Queries for each position
        tgt = self.query_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, max_length, d_model]
        
        # Single layer decoding
        out = self.decoder_layer(tgt, memory)
        
        # Project to vocabulary
        logits = self.fc_out(out)  # [batch, max_length, vocab_size]
        
        return logits

class SimpleGeneEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Initial projection layer
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Self-attention layer
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Project input to hidden dimension
        x = self.input_projection(x)
        x = self.norm1(x)
        
        # Reshape for self-attention
        # Add a sequence dimension (treating the entire gene vector as a single token)
        x = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Self-attention
        attn_output, _ = self.self_attention(x, x, x)
        
        # Reshape and project to output dimension
        x = attn_output.squeeze(1)  # [batch_size, hidden_dim]
        x = self.dropout(x)
        x = self.output_projection(x)
        
        return x
        
class FoundationModel(nn.Module):
    def __init__(self, 
                 gene_dim, 
                 tcr_vocab_size, 
                 transformer_d_model=256, 
                 tcr_output_dim=128,
                 hidden_dim=512, 
                 foundation_output_dim=256, 
                 max_length=30, 
                 dropout=0.1,
                 pad_idx=0):
        super().__init__()
        self.gene_encoder = SimpleGeneEncoder(
            input_dim=gene_dim,
            hidden_dim=hidden_dim,
            output_dim=tcr_output_dim,
            num_heads=4,
            dropout=dropout
        )
        
        self.tcr_encoder_a = TCRTransformerEncoder(
            vocab_size=tcr_vocab_size, 
            d_model=transformer_d_model, 
            nhead=8, 
            num_layers=3, 
            max_length=max_length, 
            output_dim=tcr_output_dim, 
            dropout=dropout, 
            pad_idx=pad_idx
        )
        self.tcr_encoder_b = TCRTransformerEncoder(
            vocab_size=tcr_vocab_size, 
            d_model=transformer_d_model, 
            nhead=8, 
            num_layers=3, 
            max_length=max_length, 
            output_dim=tcr_output_dim, 
            dropout=dropout, 
            pad_idx=pad_idx
        )
        self.norm_gene = nn.LayerNorm(tcr_output_dim)
        self.norm_tcr_a = nn.LayerNorm(tcr_output_dim)
        self.norm_tcr_b = nn.LayerNorm(tcr_output_dim)
        self.fusion = nn.Sequential(
            nn.Linear(tcr_output_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, foundation_output_dim),
            nn.LayerNorm(foundation_output_dim)
        )
        
    def forward(self, genes, cdr3a, cdr3b):
        gene_feat = self.gene_encoder(genes)
        tcr_a_feat = self.tcr_encoder_a(cdr3a)
        tcr_b_feat = self.tcr_encoder_b(cdr3b)
        gene_feat = self.norm_gene(gene_feat)
        tcr_a_feat = self.norm_tcr_a(tcr_a_feat)
        tcr_b_feat = self.norm_tcr_b(tcr_b_feat)
        combined = torch.cat([gene_feat, tcr_a_feat, tcr_b_feat], dim=1)
        fused_feat = self.fusion(combined)
        return gene_feat, tcr_a_feat, tcr_b_feat, fused_feat

# ----------------------------
# 3. Self-supervised Pretraining
# ----------------------------
def contrastive_loss(featuresA, featuresB, temperature=0.07):
    A = F.normalize(featuresA, dim=1)
    B = F.normalize(featuresB, dim=1)
    sim_matrix = torch.matmul(A, B.t()) / temperature
    labels = torch.arange(A.size(0), device=A.device)
    loss_AB = F.cross_entropy(sim_matrix, labels)
    loss_BA = F.cross_entropy(sim_matrix.t(), labels)
    return 0.5 * (loss_AB + loss_BA)

class MaskedPretrainingModel(nn.Module):
    def __init__(self, foundation_model, gene_dim, max_length, tcr_vocab_size, 
                 contrastive_alpha=1.0, tcr_loss_weight=1.0):
        super().__init__()
        self.foundation = foundation_model
        self.contrastive_alpha = contrastive_alpha
        self.tcr_loss_weight = tcr_loss_weight
        self.gene_decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, gene_dim)
        )
        self.gene_mask_token = nn.Parameter(torch.randn(gene_dim) * 0.02)
        self.tcr_decoder_a = TCRTransformerDecoder(
            latent_dim=256,
            d_model=256,
            max_length=max_length,
            vocab_size=tcr_vocab_size,
            nhead=4,  # Use default
            dropout=0.1
        )
        self.tcr_decoder_b = TCRTransformerDecoder(
            latent_dim=256,
            d_model=256,
            max_length=max_length,
            vocab_size=tcr_vocab_size,
            nhead=4,  # Use default
            dropout=0.1
        )
        
    def forward(self, batch, gene_mask_ratio=0.35):
        batch_size = batch['genes'].shape[0]
        mask = torch.rand(batch['genes'].shape, device=batch['genes'].device) < gene_mask_ratio
        masked_genes = batch['genes'].clone()
        
        # Use learned mask token instead of zeros
        mask_token_expanded = self.gene_mask_token.unsqueeze(0).expand(batch_size, -1)
        masked_genes[mask] = mask_token_expanded.expand_as(masked_genes)[mask]
        gene_feat, tcr_a_feat, tcr_b_feat, fused_feat = self.foundation(
            masked_genes, 
            batch['cdr3a'], 
            batch['cdr3b']
        )
        gene_recon = self.gene_decoder(fused_feat)
        gene_loss = F.mse_loss(gene_recon[mask], batch['genes'][mask])
        logits_a = self.tcr_decoder_a(fused_feat)
        logits_b = self.tcr_decoder_b(fused_feat)
        vocab_size = logits_a.size(-1)
        logits_a_flat = logits_a.view(-1, vocab_size)
        logits_b_flat = logits_b.view(-1, vocab_size)
        targets_a = batch['cdr3a'].view(-1)
        targets_b = batch['cdr3b'].view(-1)
        tcr_loss_a = F.cross_entropy(logits_a_flat, targets_a, ignore_index=0)
        tcr_loss_b = F.cross_entropy(logits_b_flat, targets_b, ignore_index=0)
        tcr_loss = self.tcr_loss_weight * 0.5 * (tcr_loss_a + tcr_loss_b)
        c_loss = 0.5 * contrastive_loss(gene_feat, tcr_a_feat, temperature=0.07) + \
                 0.5 * contrastive_loss(gene_feat, tcr_b_feat, temperature=0.07)
        total_loss = gene_loss + tcr_loss + self.contrastive_alpha * c_loss
        return total_loss, gene_loss, tcr_loss, c_loss
        
def load_foundation_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = FoundationModel(
        gene_dim=checkpoint['gene_dim'],
        tcr_vocab_size=checkpoint['tcr_vocab_size']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['vocab']
    
# ----------------------------
# Revised Training Loop
# ----------------------------
def train(adata, epochs=100, batch_size=2048, save_dir='models', 
          accumulation_steps=4, val_ratio=0.1, early_stopping_patience=50,
          max_length=30, add_special_tokens=True,
          warmup_epochs=10, improvement_threshold=1e-4):
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    
    os.makedirs(save_dir, exist_ok=True)
    dataset = SingleCellDataset(
        adata, 
        max_length=max_length, 
        add_special_tokens=add_special_tokens
    )
    
    train_size = int((1 - val_ratio) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    foundation = FoundationModel(
        gene_dim=adata.n_vars,
        tcr_vocab_size=len(dataset.tcr_vocab.vocab),
        max_length=max_length,
        dropout=0.1
    )
    model = MaskedPretrainingModel(
        foundation,
        gene_dim=adata.n_vars,
        max_length=max_length,
        tcr_vocab_size=len(dataset.tcr_vocab.vocab),
        tcr_loss_weight=1.0
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Combined LR scheduler with warmup and cosine decay
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        else:
            # cosine decay after warmup
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    gradient_clip_norm = 1.0  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Model size: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    print(f"Using device: {device}")
    
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = {
        'train_loss': [],
        'val_loss': [],
        'gene_loss': [],
        'tcr_loss': [],
        'contrastive_loss': []
    }
    
    best_val_loss = float('inf')
    early_stopping_counter = 0
    best_epoch = 0
    total_steps = epochs * len(train_loader)
    pbar = tqdm(total=total_steps, desc="Training Progress")
    first_epoch_start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_gene_loss = 0.0
        epoch_tcr_loss = 0.0
        epoch_c_loss = 0.0
        
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, gene_loss, tcr_loss, c_loss = model(batch)
                loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            epoch_loss += loss.item() * accumulation_steps
            epoch_gene_loss += gene_loss.item()
            epoch_tcr_loss += tcr_loss.item()
            epoch_c_loss += c_loss.item()
            pbar.set_postfix({'Loss': f'{loss.item() * accumulation_steps:.4f}'})
            pbar.update(1)
        
        if epoch == 0:
            first_epoch_end_time = time.time()
            print(f"Time for first epoch: {first_epoch_end_time - first_epoch_start_time:.2f} seconds")

        avg_train_loss = epoch_loss / len(train_loader)
        avg_gene_loss = epoch_gene_loss / len(train_loader)
        avg_tcr_loss = epoch_tcr_loss / len(train_loader)
        avg_c_loss = epoch_c_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.cuda.amp.autocast(enabled=use_amp):
                    loss, _, _, _ = model(batch)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step()
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['gene_loss'].append(avg_gene_loss)
        history['tcr_loss'].append(avg_tcr_loss)
        history['contrastive_loss'].append(avg_c_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch [{epoch + 1}/{epochs}] - '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, '
                  f'Gene Loss: {avg_gene_loss:.4f}, '
                  f'TCR Loss: {avg_tcr_loss:.4f}, '
                  f'Contrastive Loss: {avg_c_loss:.4f}, '
                  f'LR: {scheduler.get_last_lr()[0]:.6f}')

        # Check for significant improvement
        if best_val_loss - avg_val_loss > improvement_threshold:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            early_stopping_counter = 0
            checkpoint_path = os.path.join(save_dir, 'foundation_model_best.pt')
            checkpoint = {
                'model_state_dict': model.foundation.state_dict(),
                'gene_decoder_state_dict': model.gene_decoder.state_dict(),
                'vocab': dataset.tcr_vocab.vocab,
                'epoch': epoch + 1,
                'loss': avg_val_loss,
                'gene_dim': adata.n_vars,
                'tcr_vocab_size': len(dataset.tcr_vocab.vocab),
                'max_length': max_length,
                'special_tokens': dataset.tcr_vocab.special_tokens
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved best model checkpoint to {checkpoint_path}")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs. Best epoch: {best_epoch}")
                break

        if (epoch + 1) % 50 == 0:
            checkpoint_path = os.path.join(save_dir, f'foundation_model_epoch_{epoch+1}.pt')
            checkpoint = {
                'model_state_dict': model.foundation.state_dict(),
                'gene_decoder_state_dict': model.gene_decoder.state_dict(),
                'vocab': dataset.tcr_vocab.vocab,
                'epoch': epoch + 1,
                'loss': avg_train_loss,
                'gene_dim': adata.n_vars,
                'tcr_vocab_size': len(dataset.tcr_vocab.vocab),
                'max_length': max_length,
                'special_tokens': dataset.tcr_vocab.special_tokens
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved model checkpoint to {checkpoint_path}")

    pbar.close()

    best_model_path = os.path.join(save_dir, 'foundation_model_best.pt')
    best_model, _ = load_foundation_model(best_model_path)
    return best_model, history