import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

try: 
    from .mae import MAE_ViT, MAE_Encoder
except ImportError:
    from mae import MAE_ViT, MAE_Encoder

class MaskingModel(nn.Module):
    def __init__(self, pretrained_mae: MAE_ViT, mask_predictor, use_gumbel_softmax=False):
        super().__init__()

        # Create deterministic encoder from pretrained weights
        self.encoder = DeterministicMAEEncoder(pretrained_mae.encoder)
        self.decoder = pretrained_mae.decoder

        # Freeze encoder and decoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False

        self.encoder.eval()
        self.decoder.eval()

        # Only mask_predictor is trainable
        self.mask_predictor = mask_predictor
        self.use_gumbel_softmax = use_gumbel_softmax

    def forward(self, x, tau=0.5):
        features, mask_probs, backward_indexes = self.get_features_and_masks(x, tau=tau)

        # Exclude class token from features and mask_probs
        cls_token = features[:, :1, :]  # [B, 1, C]
        features_without_cls = features[:, 1:, :]  # [B, num_patches, C]

        # Ensure mask_probs matches features_without_cls
        assert mask_probs.shape == features_without_cls.shape[:2], "Mismatch in mask_probs and features shape"

        # Get the mask token and expand it to match patch features
        mask_token = self.decoder.mask_token  # [1, 1, C]
        B, num_patches, C = features_without_cls.shape
        mask_token_expanded = mask_token.expand(B, num_patches, C)  # [B, num_patches, C]

        # Compute the new masked features by interpolating between original features and mask token
        mask_probs_unsq = mask_probs.unsqueeze(-1)  # [B, num_patches, 1]
        masked_features = mask_probs_unsq * features_without_cls + (1 - mask_probs_unsq) * mask_token_expanded  # [B, num_patches, C]

        # Reconstruct features by concatenating the class token
        masked_features_with_cls = torch.cat([cls_token, masked_features], dim=1)  # [B, num_patches+1, C]

        # Use backward_indexes from the encoder
        reconstructed_img, _ = self.decoder(masked_features_with_cls, backward_indexes)

        return reconstructed_img, mask_probs

    def get_features_and_masks(self, x, tau=0.5):
        # Get mask logits from the mask predictor
        mask_logits = self.mask_predictor(x)  # Shape: [B, num_patches]

        # Apply sigmoid to get mask probabilities
        mask_probs = F.sigmoid(mask_logits)  # [B, num_patches]

        if self.use_gumbel_softmax:
            # apply gumbel softmax to mask probabilities
            rand = torch.rand_like(mask_probs)
            mask_probs = F.sigmoid(-(torch.log(mask_probs / (1 - mask_probs)) + torch.log(rand / (1 - rand))) / tau)

        # Encode the image
        features, backward_indexes = self.encoder(x)  # features: [B, T, C], backward_indexes: [B, T]

        return features, mask_probs, backward_indexes

class DeterministicMAEEncoder(nn.Module):
    """MAE Encoder without random shuffling - preserves patch order"""
    def __init__(self, pretrained_encoder: MAE_Encoder):
        super().__init__()
        self.cls_token = pretrained_encoder.cls_token
        self.pos_embedding = pretrained_encoder.pos_embedding
        self.patchify = pretrained_encoder.patchify
        self.transformer = pretrained_encoder.transformer
        self.layer_norm = pretrained_encoder.layer_norm
        self.num_patches = pretrained_encoder.num_patches
    
    def forward(self, img):
        """
        Forward pass without shuffling - maintains original patch order
        Returns identity backward_indexes
        """
        B = img.shape[0]
        
        # Patchify and add positional embeddings
        patches = self.patchify(img)  # (B, emb_dim, h, w)
        patches = rearrange(patches, 'b c h w -> b (h w) c')  # (B, num_patches, C)
        patches = patches + self.pos_embedding[:, 1:, :]  # Add pos embeddings
        
        # Prepend cls token (no shuffling)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        patches = torch.cat([cls_tokens, patches], dim=1)  # (B, num_patches+1, C)
        
        # Transformer layers
        features = self.transformer(patches)
        features = self.layer_norm(features)
        
        # Return identity backward_indexes (no shuffling occurred)
        backward_indexes = torch.arange(self.num_patches, device=img.device).unsqueeze(0).expand(B, -1)
        
        return features, backward_indexes
