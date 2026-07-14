import torch
import torch.nn as nn
import torch.nn.functional as F

class TabularAttentionNetwork(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super(TabularAttentionNetwork, self).__init__()
        
        # We embed the tabular features into a higher-dimensional space
        self.embedding = nn.Linear(input_dim, 128)
        
        # Self-Attention Mechanism to learn relationships between different physical features
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        
        # Feed-forward network
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        x = F.relu(self.embedding(x))
        
        # To use MultiheadAttention, we add a sequence dimension (batch_size, seq_len=1, embed_dim)
        x = x.unsqueeze(1)
        
        # Self-attention applied across the embedded features
        attn_out, _ = self.attention(x, x, x)
        
        # Squeeze back to (batch_size, embed_dim)
        x = attn_out.squeeze(1)
        
        # Feed-forward layers
        latent_embedding = F.relu(self.fc1(x)) # We extract this layer for UMAP!
        x = self.dropout(latent_embedding)
        logits = self.fc2(x)
        
        return logits, latent_embedding

def physics_informed_loss(logits, targets, inputs, feature_names):
    """
    Custom Loss Function that penalizes physically impossible predictions.
    Standard Cross-Entropy is combined with a Physics Penalty.
    """
    base_loss = F.cross_entropy(logits, targets)
    
    # Enforce Transit Depth Geometry: Depth ≈ (R_planet / R_star)^2
    # We penalize the model if it predicts CONFIRMED but the physical geometry is impossible.
    try:
        # Get column indices dynamically from the one-hot encoded dataset
        idx_depth = feature_names.get_loc('koi_depth')
        idx_prad = feature_names.get_loc('koi_prad')
        idx_srad = feature_names.get_loc('koi_srad')
        
        depth_obs = inputs[:, idx_depth]
        # Calculate theoretical depth (add epsilon to prevent division by zero)
        depth_calc = (inputs[:, idx_prad] / (inputs[:, idx_srad] + 1e-5)) ** 2
        
        # Calculate the magnitude of the physical violation
        physics_violation = F.mse_loss(depth_obs, depth_calc, reduction='none')
        
        # We only apply the penalty if the model is predicting CONFIRMED
        preds = torch.argmax(logits, dim=1)
        physics_penalty = (physics_violation * (preds == 1).float()).mean()
    except KeyError:
        # Graceful fallback if features were pruned during EDA
        physics_penalty = torch.tensor(0.0, device=logits.device)
    
    # Lambda hyperparameter tuned via ablation study
    lambda_phys = 0.1
    total_loss = base_loss + (lambda_phys * physics_penalty)
    
    return total_loss
