import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

# Implement trunc_normal_ function
def trunc_normal_(tensor, mean=0.0, std=1.0):
    # Truncated normal distribution between [mean - 2*std, mean + 2*std]
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor

# Define MLP class
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=(0.0, 0.0)):
        assert len(drop) == 2, "drop must be a tuple of two elements"
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop[1])
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

# Define Attention class
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0, fused_attn=False):
        super().__init__()
        assert dim % num_heads == 0, f"Embedding dimension {dim} should be divisible by number of heads {num_heads}."
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5  # Scaling factor
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, C = x.shape  # x: (batch_size, seq_len, embed_dim)
        qkv = self.qkv(x)  # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)  # (B, N, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv.unbind(dim=0)  # Each: (B, num_heads, N, head_dim)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
            attn = attn.softmax(dim=-1)  # Apply softmax over the last dimension
            attn = self.attn_drop(attn)
            x = (attn @ v)  # (B, num_heads, N, head_dim)

        x = x.transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# Define Transformer Block
class Block(nn.Module):
    """
    A block consists of a multi-head attention layer and a MLP layer.

    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0.0,
                 attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, fused_attn=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, fused_attn=fused_attn)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=(drop, drop))
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))  # Residual connection
        x = x + self.mlp(self.norm2(x))   # Residual connection
        return x

def random_indexes(size: int):
    perm = torch.randperm(size)  # shape (size,)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(size)
    return perm, inv

def take_indexes(sequences, indexes):
    """
    Gather elements from sequences using indexes.
    
    Args:
        sequences: (B, T, C) tensor
        indexes: (B, T') tensor of indices
    
    Returns:
        gathered: (B, T', C) tensor
    """
    return torch.gather(sequences, 1, repeat(indexes, 'b t -> b t c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches: torch.Tensor):
        """
        Args:
            patches: (B, T, C) tensor
            
        Returns:
            patches: (B, remain_T, C) tensor
            forward_indexes: (B, T) tensor
            backward_indexes: (B, T) tensor
        """
        B, T, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.stack([i[0] for i in indexes], dim=0).to(patches.device)  # (B, T)
        backward_indexes = torch.stack([i[1] for i in indexes], dim=0).to(patches.device)  # (B, T)
        
        patches = take_indexes(patches, forward_indexes)  # (B, T, C)
        patches = patches[:, :remain_T, :]  # (B, remain_T, C)

        return patches, forward_indexes, backward_indexes

class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=12,
                 num_head=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.num_patches = (image_size // patch_size) ** 2
        # pos_embedding is now (1, num_patches + 1, emb_dim) for batch-first
        self.pos_embedding = torch.nn.Parameter(torch.zeros(1, self.num_patches + 1, emb_dim), requires_grad=False)
        self.shuffle = PatchShuffle(mask_ratio)

        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head, fused_attn=True) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embedding.shape[-1], int(self.num_patches**.5), cls_token=True)
        # Change unsqueeze to dim 0 for (1, num_patches+1, emb_dim)
        self.pos_embedding.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize the conv2d layer as a linear layer, consistent with FAIR implementation
        w = self.patchify.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, img):
        """
        Args:
            img: (B, 3, H, W)
            
        Returns:
            features: (B, remain_T+1, C) - includes cls token
            backward_indexes: (B, num_patches)
        """
        patches = self.patchify(img)  # Shape: (B, emb_dim, h, w)
        patches = rearrange(patches, 'b c h w -> b (h w) c')  # Shape: (B, num_patches, emb_dim)
        patches = patches + self.pos_embedding[:, 1:, :]  # Add positional embeddings (excluding cls)

        patches, forward_indexes, backward_indexes = self.shuffle(patches)  # (B, remain_T, C)

        # Prepend cls token
        cls_tokens = self.cls_token.expand(patches.shape[0], -1, -1)  # (B, 1, C)
        patches = torch.cat([cls_tokens, patches], dim=1)  # (B, remain_T+1, C)
        
        features = self.transformer(patches)  # (B, remain_T+1, C)
        features = self.layer_norm(features)

        return features, backward_indexes

class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 enc_emb_dim=192,
                 dec_emb_dim=192,
                 num_layer=4,
                 num_head=3,
                 ) -> None:
        super().__init__()

        self.proj_dec_emb = torch.nn.Linear(enc_emb_dim, dec_emb_dim)
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, dec_emb_dim))
        self.num_patches = (image_size // patch_size) ** 2
        # pos_embedding is now (1, num_patches + 1, emb_dim) for batch-first
        self.pos_embedding = torch.nn.Parameter(torch.zeros(1, self.num_patches + 1, dec_emb_dim), requires_grad=False)

        self.transformer = torch.nn.Sequential(*[Block(dec_emb_dim, num_head, fused_attn=True) for _ in range(num_layer)])

        self.head = torch.nn.Linear(dec_emb_dim, 3 * patch_size ** 2)
        self.patch2img = Rearrange('b (h w) (c p1 p2) -> b c (h p1) (w p2)',
                                   p1=patch_size, p2=patch_size, h=image_size // patch_size)
        self.patch_size = patch_size
        self.image_size = image_size

        self.initialize_weights()

    def initialize_weights(self):
        # initialize the positional embeddings
        pos_embed = get_2d_sincos_pos_embed(self.pos_embedding.shape[-1], int(self.num_patches**.5), cls_token=True)
        self.pos_embedding.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize the mask token
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, features, backward_indexes):
        """
        Args:
            features: (B, remain_T+1, C) - includes cls token
            backward_indexes: (B, num_patches)
            
        Returns:
            img: (B, 3, H, W)
            mask: (B, 3, H, W) - binary mask indicating which patches were masked
        """
        B = features.shape[0]
        T = features.shape[1]  # remain_T + 1 (includes cls)
        total_patches = self.pos_embedding.shape[1]  # num_patches + 1 (for cls token)

        # Create backward indexes with cls token at position 0
        # backward_indexes is (B, num_patches), we need to add cls and shift indices
        backward_indexes_with_cls = torch.cat([
            torch.zeros(B, 1, dtype=torch.long, device=backward_indexes.device), 
            backward_indexes + 1
        ], dim=1)  # (B, num_patches+1)

        features = self.proj_dec_emb(features)
        
        # Pad features with mask tokens to reconstruct the full sequence
        num_masked = total_patches - T
        mask_tokens = self.mask_token.expand(B, num_masked, -1)  # (B, num_masked, C)
        features = torch.cat([features, mask_tokens], dim=1)  # (B, num_patches+1, C)
        
        # Reorder features using backward indexes
        features = take_indexes(features, backward_indexes_with_cls)  # (B, num_patches+1, C)
        features = features + self.pos_embedding  # Add positional embeddings
        
        features = self.transformer(features)  # (B, num_patches+1, C)
        features = features[:, 1:, :]  # Remove cls token -> (B, num_patches, C)
        
        # Generate patches
        patches = self.head(features)  # (B, num_patches, patch_size^2 * 3)
        
        # Create mask: 1 for masked patches, 0 for visible patches
        # The first T-1 patches are visible (T includes cls, so T-1 = remain_T)
        # The remaining patches are masked
        mask = torch.zeros_like(patches)
        mask[:, T-1:, :] = 1  # Mark patches that were masked
        
        # Reorder mask to match original patch order
        mask = take_indexes(mask, backward_indexes)  # (B, num_patches, C)
        
        # Convert patches and mask back to image
        reconstructed_img = self.patch2img(patches)
        mask = self.patch2img(mask)

        return reconstructed_img, mask

class MAE_ViT(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 enc_emb_dim=192,
                 dec_emb_dim=512,
                 encoder_layer=12,
                 encoder_head=3,
                 decoder_layer=8,
                 decoder_head=16,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()
            
        self.encoder = MAE_Encoder(image_size, patch_size, enc_emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.decoder = MAE_Decoder(image_size, patch_size, enc_emb_dim, dec_emb_dim, decoder_layer, decoder_head)

    def forward(self, img):
        features, backward_indexes = self.encoder(img)
        predicted_img, mask = self.decoder(features, backward_indexes)
        return predicted_img, mask                 
