import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import numpy as np
from einops import rearrange

def plot_hyperparameter_impact(results):
    """Plot the impact of different hyperparameters on model performance"""
    # Convert results list to DataFrame
    df = pd.DataFrame(results)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Impact of Hyperparameters on Model Performance')
    
    # Plot for each hyperparameter
    sns.boxplot(data=df, x='embed_dim', y='accuracy', ax=axes[0,0])
    axes[0,0].set_title('Embedding Dimension Impact')
    
    sns.boxplot(data=df, x='num_heads', y='accuracy', ax=axes[0,1])
    axes[0,1].set_title('Number of Heads Impact')
    
    sns.boxplot(data=df, x='num_layers', y='accuracy', ax=axes[0,2])
    axes[0,2].set_title('Number of Layers Impact')
    
    sns.boxplot(data=df, x='dropout', y='accuracy', ax=axes[1,0])
    axes[1,0].set_title('Dropout Impact')
    
    sns.boxplot(data=df, x='pool', y='accuracy', ax=axes[1,1])
    axes[1,1].set_title('Pooling Strategy Impact')
    
    # Learning rate requires different scaling due to small values
    sns.boxplot(data=df, x='lr', y='accuracy', ax=axes[1,2])
    axes[1,2].set_title('Learning Rate Impact')
    
    plt.tight_layout()
    plt.savefig('hyperparameter_impact.png')
    plt.close()

def get_attention_maps(model, img):
    """Extract attention maps from the model"""
    attention_maps = []
    
    def hook_fn(module, input, output):
        # Get attention weights from the output
        attention = module.attention  # Get the attention matrix
        # Reshape from (batch_size * num_heads, seq_len, seq_len) to (batch_size, num_heads, seq_len, seq_len)
        batch_size = img.size(0)
        attention = attention.view(batch_size, module.num_heads, -1, attention.size(-1))
        # Average over heads
        attention = attention.mean(dim=1)  # Average over heads
        attention_maps.append(attention.detach())
    
    # Register hooks for all attention layers
    hooks = []
    for block in model.transformer_blocks:
        hooks.append(block.attention.register_forward_hook(hook_fn))
    
    # Move image to same device as model
    device = next(model.parameters()).device
    img = img.to(device)
    
    # Forward pass
    with torch.no_grad():
        model(img)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return attention_maps

def visualize_attention(model, img, save_path='attention_maps.png'):
    """Visualize attention maps for a given image"""
    # Move image to same device as model
    device = next(model.parameters()).device
    img = img.to(device)
    
    # Get attention maps
    attention_maps = get_attention_maps(model, img.unsqueeze(0))
    
    # Number of attention layers
    num_layers = len(attention_maps)
    
    # Create figure
    fig, axes = plt.subplots(2, num_layers//2, figsize=(20, 8))
    fig.suptitle('Attention Maps Across Layers')
    
    # Calculate patch size
    H, W = img.shape[-2:]
    patch_size = model.patch_embedding[0].patch_height  # Get patch size from model
    n_patches_h = H // patch_size
    n_patches_w = W // patch_size
    
    for i, attn_map in enumerate(attention_maps):
        # Get attention weights for first head of first image
        attn = attn_map[0]  # First image
        
        # Calculate actual dimensions
        if model.pool == 'cls':
            # Remove cls token attention
            attn = attn[1:, 1:]
        
        # Reshape attention map to match patch grid
        attn = attn.reshape(n_patches_h, n_patches_w, n_patches_h, n_patches_w)
        # Average attention across query patches
        attn = attn.mean(dim=(0, 1)).cpu()
        
        # Plot
        row, col = i // (num_layers//2), i % (num_layers//2)
        if isinstance(axes, np.ndarray):
            ax = axes[row, col]
        else:
            ax = axes  # In case we only have one layer
        im = ax.imshow(attn, cmap='viridis')
        ax.set_title(f'Layer {i+1}')
        ax.axis('off')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Add this to your main training loop
def analyze_model(model, results, test_image):
    """Analyze model performance and attention patterns"""
    # Plot hyperparameter impact
    plot_hyperparameter_impact(results)
    
    # Visualize attention maps
    visualize_attention(model, test_image)
    
    # Print best configuration
    best_result = max(results, key=lambda x: x['accuracy'])
    print("\nBest Configuration:")
    for k, v in best_result.items():
        print(f"{k}: {v}") 