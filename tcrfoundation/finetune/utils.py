import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


# After computing UMAPs and creating individual plots
def create_combined_visualization(adata, label_column, results_dir):
    """
    Create a combined visualization of all UMAPs and save it to the results directory.
    Also removes individual plots after creating the combined figure.
    """
    # First compute UMAPs if not already done
    adata = compute_umaps(adata)
    
    # Create individual visualizations
    embeddings = ['X_gene_umap', 'X_fused_emb_umap', 'X_gene_emb_umap', 
                  'X_tcr_a_emb_umap', 'X_tcr_b_emb_umap']
    
    # Create res directory if it doesn't exist
    os.makedirs('res', exist_ok=True)
    
    # Generate individual plots
    for emb in embeddings:
        from plot_funcs import plot_dimensionality_reduction
        plot_dimensionality_reduction(adata, emb, label_column)
    
    # Create combined visualization
    embedding_names = ['gene', 'fused_emb', 'gene_emb', 'tcr_a_emb', 'tcr_b_emb']
    image_paths = [f'res/{label_column}_on_{emb.lower()}_umap.png' 
                   for emb in embedding_names]
    
    # Plot combined figure
    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    ax = ax.flatten()
    
    for i, img_path in enumerate(image_paths):
        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            ax[i].imshow(img)
            ax[i].axis('off')
    
    # Hide the last subplot if it's empty
    if len(image_paths) < len(ax):
        for i in range(len(image_paths), len(ax)):
            ax[i].set_visible(False)
    
    plt.tight_layout()
    combined_path = f'{results_dir}/{label_column}_combined_umap.png'
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created combined visualization: {combined_path}")
    
    # Delete individual figures after creating the combined one
    for img_path in image_paths:
        if os.path.exists(img_path):
            os.remove(img_path)
            print(f"Deleted individual plot: {img_path}")
            
def compute_umaps(adata, n_neighbors=15, random_state=42):
    """
    Compute UMAP embeddings for multiple representations in an AnnData object.

    For each representation in:
      - 'X_fused_emb'
      - 'X_gene_emb'
      - 'X_tcr_a_emb'
      - 'X_tcr_b_emb'
    
    this function computes a neighborhood graph using that representation and then
    computes the UMAP embedding. The result is stored in adata.obsm with keys:
      - 'X_umap_X_fused_emb'
      - 'X_umap_X_gene_emb'
      - 'X_umap_X_tcr_a_emb'
      - 'X_umap_X_tcr_b_emb'
    
    Parameters
    ----------
    adata : AnnData
        The annotated data matrix.
    n_neighbors : int, default 15
        The number of neighbors to use when computing the neighborhood graph.
    
    Returns
    -------
    adata : AnnData
        The same AnnData object with added UMAP embeddings in the .obsm attribute.
    """
    import random
    import numpy as np
    
    random.seed(random_state)
    np.random.seed(random_state)
    representations = ['X_fused_emb', 'X_gene_emb', 'X_tcr_a_emb', 'X_tcr_b_emb', 'X_gene']
    
    for rep in representations:
        if rep == 'X_gene':
            sc.pp.neighbors(adata, n_neighbors=n_neighbors)
        
            # Compute the UMAP embedding from the neighbor graph.
            sc.tl.umap(adata)
            
            # Save the computed UMAP embedding to a new key in adata.obsm.
            adata.obsm[f"{rep}_umap"] = adata.obsm['X_umap'].copy()
        else:
            # Compute the neighborhood graph using the specified representation.
            # Here, we assume that adata.obsm[rep] exists and contains the relevant embedding.
            sc.pp.neighbors(adata, use_rep=rep, n_neighbors=n_neighbors)
            
            # Compute the UMAP embedding from the neighbor graph.
            sc.tl.umap(adata)
            
            # Save the computed UMAP embedding to a new key in adata.obsm.
            adata.obsm[f"{rep}_umap"] = adata.obsm['X_umap'].copy()
    
    return adata
    
def plot_dimensionality_reduction(adata, emb_key='X_umap', color_key='binding_name', 
                                figsize=(8, 5), 
                                dpi=300, 
                                point_size=30, 
                                alpha=1.0):
    """
    Create a visualization of dimensionality reduction results with categorical colors
    """
    
    # Shuffle the order of cells for better visualization
    shuffle_idx = np.random.permutation(adata.shape[0])
    emb_coords = adata.obsm[emb_key][shuffle_idx]
    color_labels = adata.obs[color_key].iloc[shuffle_idx]
    
    # Create figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    # Create main plot with fixed position
    ax = plt.axes([0.1, 0.1, 0.6, 0.8])  # [left, bottom, width, height]
    
    # Get unique categories and create a more diverse color palette
    categories = adata.obs[color_key].unique()
    n_colors = len(categories)
    
    # Create a more diverse color palette by combining different color spaces
    base_colors = (
        sns.color_palette("Set2", n_colors=min(8, n_colors)) +  # Muted but distinct
        sns.color_palette("Set1", n_colors=min(9, n_colors)) +  # Bright and distinct
        sns.color_palette("Dark2", n_colors=min(8, n_colors)) +  # Dark and distinct
        sns.color_palette("Paired", n_colors=min(12, n_colors))  # Paired colors
    )
    palette = base_colors[:n_colors]  # Take only as many colors as we need
    
    # Create scatter plot for each category
    for idx, category in enumerate(categories):
        mask = color_labels == category
        ax.scatter(emb_coords[mask, 0],
                  emb_coords[mask, 1],
                  c=[palette[idx]],
                  label=category,
                  alpha=alpha,
                  s=point_size,
                  edgecolor='none')
    
    # Customize the plot
    ax.set_xlabel(f'{emb_key.replace("X_", "").upper()} 1', fontsize=12)
    ax.set_ylabel(f'{emb_key.replace("X_", "").upper()} 2', fontsize=12)
    ax.set_title(f'{color_key} on {emb_key.replace("X_", "").upper()}', fontsize=14, pad=20)
    
    # Remove ticks but keep lines
    ax.tick_params(axis='both', which='both', length=0)
    
    # Add legend with fixed position
    legend = ax.legend(bbox_to_anchor=(1.05, 0.5),
                      loc='center left',
                      frameon=True,
                      fancybox=True,
                      shadow=True,
                      fontsize=10)
    
    # Save plot
    plt.savefig(f'res/{color_key}_on_{emb_key.replace("X_", "").lower()}.png', 
                dpi=dpi, 
                bbox_inches='tight')
    
    return fig, ax
    
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    Create a new colormap from a subset of the original colormap.
    """
    new_colors = cmap(np.linspace(minval, maxval, n))
    new_cmap = LinearSegmentedColormap.from_list(
        f"truncated({cmap.name},{minval:.2f},{maxval:.2f})", new_colors
    )
    return new_cmap

def plot_all_metrics_bubble_chart(df, results_dir, label='tissue'):
    """
    Create three bubble charts side by side for all metrics with data-driven text coloring,
    using a unified bubble size scale across subplots.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.colors as colors
    
    # Mapping dictionary for mode names
    mode_mapping = {
        'rna_only': 'RNA only',
        'tcr_only': 'TCR α+β',
        'tcra_only': 'TCR α only',
        'tcrb_only': 'TCR β only',
        'rna_tcr': 'RNA+TCR'
    }
    
    # Create a copy of the dataframe with renamed modes
    df_plot = df.copy()
    df_plot['Mode'] = df_plot['Mode'].map(mode_mapping)
    
    # Define metrics to plot
    metrics = ['Accuracy', 'F1 Weighted', 'F1 Macro']
    
    # Compute global min and max for all metrics to use for unified bubble scaling
    global_min = df_plot[metrics].min().min()
    global_max = df_plot[metrics].max().max()
    
    # Define function to truncate colormap
    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = colors.LinearSegmentedColormap.from_list(
            f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap
    
    # Optionally, if you want to use only the middle part of 'mako' for the hue, uncomment:
    truncated_p = truncate_colormap(plt.get_cmap("summer_r"), 0.3, 1.0)
    palette_used = truncated_p
    # Otherwise, simply use the full 'mako' palette:
    # palette_used = plt.get_cmap("mako")
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.0), dpi=300)
    
    for ax, metric in zip(axes, metrics):
        # Calculate threshold for text color using the median value for the current metric
        threshold = df_plot[metric].median() * 0.99
        
        # Create scatter plot with unified size normalization across subplots
        sns.scatterplot(
            data=df_plot,
            x='Split',
            y='Mode',
            size=metric,
            hue=metric,
            palette=palette_used,
            sizes=(800, 2000),
            size_norm=plt.Normalize(global_min, global_max),
            legend=False,
            ax=ax
        )
        
        # Add text annotations for each point with dynamic coloring
        for _, row in df_plot.iterrows():
            text_color = 'white' if row[metric] > threshold else 'black'
            ax.text(
                x=row['Split'],
                y=row['Mode'],
                s=f'{row[metric]:.3f}',  # Format to 3 decimal places
                horizontalalignment='center',
                verticalalignment='center',
                color=text_color,
                fontweight=500,
                fontsize=10
            )
        
        # Styling for each subplot
        ax.set_title(metric, pad=20, fontsize=10)
        ax.set_xlabel("Data Split", labelpad=15)
        ax.set_ylabel("Model Mode", labelpad=15)
        ax.grid(True, which='both', axis='both', linestyle=':', alpha=0.3)
        ax.margins(x=0.3, y=0.3)
    
    # Adjust layout and add main title
    plt.tight_layout()
    fig.suptitle(f'Metrics across different modes and splits - {label}', 
                 y=1.05, 
                 fontsize=12, 
                 fontweight=500)
    plt.savefig(f'{results_dir}/{label}_bubble.png')
    plt.show()
    
def build_results_dataframe(results):
    """
    Build a pandas DataFrame for a chosen metric (here F1 Weighted).
    Rows = each (mode, split)
    Columns = [mode, split, f1_weighted]
    """
    rows = []
    for mode, split_dict in results.items():
        for split, metrics_tuple in split_dict.items():
            acc, f1_macro, f1_weighted = metrics_tuple
            rows.append({
                'Mode': mode,
                'Split': split,
                'Accuracy': acc,
                'F1 Weighted': f1_weighted,
                'F1 Macro': f1_macro
                
            })
    df = pd.DataFrame(rows)
    return df