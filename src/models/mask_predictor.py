import torch
from torch import nn
from einops import rearrange
try: 
    from .mae import Block, get_2d_sincos_pos_embed, MLP
except ImportError:
    # if running as a standalone script
    from mae import Block, get_2d_sincos_pos_embed, MLP

class ImageTransformerMaskPredictor(nn.Module):
    def __init__(self, image_size=32, patch_size=2, in_channels=3, embed_dim=192, num_patches=None, num_layers=2, num_heads=4):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        if num_patches is None:
            self.num_patches = (image_size // patch_size) ** 2
        else:
            self.num_patches = num_patches

        # Patchify the image
        self.patchify = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Positional Embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)

        # Transformer Layers
        self.transformer = nn.Sequential(*[
            Block(embed_dim, num_heads, fused_attn=True) for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Head to output mask logits
        hidden_features = embed_dim * 4
        self.head = MLP(in_features=embed_dim, hidden_features=hidden_features, out_features=1)

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embedding.shape[-1], int(self.num_patches**.5), cls_token=False)
        self.pos_embedding.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.patchify.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: [B, C, H, W]
        B = x.shape[0]

        # Patchify the image
        patches = self.patchify(x)  # [B, embed_dim, H', W']
        patches = rearrange(patches, 'b c h w -> b (h w) c')  # [B, num_patches, embed_dim]

        # Add positional embeddings
        x = patches + self.pos_embedding  # [B, num_patches, embed_dim]

        # Process with transformer layers
        x = self.transformer(x)
        x = self.layer_norm(x)

        # Compute mask logits
        mask_logits = self.head(x).squeeze(-1)  # [B, num_patches, 1] -> [B, num_patches]

        return mask_logits