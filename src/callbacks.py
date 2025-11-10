from lightning.pytorch.callbacks import Callback
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import wandb

class ComprehensiveMaskingVisualizationCallback(Callback):
    """
    Comprehensive visualization showing the full masking and reconstruction pipeline.
    
    """
    
    def __init__(self, use_hard_masks: bool = False):
        """
        Initialize the visualization callback.
        
        Args:
            use_hard_masks: If True, applies hard thresholding at 0.5 to create binary
                           masking decisions
        """
        self.use_hard_masks = use_hard_masks
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Called at the end of each validation epoch to generate and log visualizations.
        
        """
        if not trainer.datamodule:
            return

        val_dataset = trainer.datamodule.val_dataset
        
        # Sample 16 validation images to create a 4x4 grid
        val_img_samples = torch.stack([val_dataset[i][0] for i in range(16)]).to(
            pl_module.device
        )
        
        # Run the masking model to get reconstructions and mask probabilities
        x_recon, mask_probs = pl_module(val_img_samples)
        x_recon = torch.clamp(x_recon, 0, 1)
        
        # Decide whether to use soft or hard masks based on callback configuration
        if self.use_hard_masks:
            mask_probs_used = (mask_probs > 0.5).float()
            
            features, _, backward_indexes = pl_module.masking_model.get_features_and_masks(val_img_samples)
            cls_token = features[:, :1, :]
            features_without_cls = features[:, 1:, :]
            
            # Get the learned mask token from the decoder
            mask_token = pl_module.masking_model.decoder.mask_token
            B, num_patches, C = features_without_cls.shape
            mask_token_expanded = mask_token.expand(B, num_patches, C)
            
            mask_probs_hard_unsq = mask_probs_used.unsqueeze(-1)
            masked_features_hard = (
                mask_probs_hard_unsq * features_without_cls + 
                (1 - mask_probs_hard_unsq) * mask_token_expanded
            )
            
            # Reconstruct the image with hard-masked features
            masked_features_with_cls = torch.cat([cls_token, masked_features_hard], dim=1)
            x_recon, _ = pl_module.masking_model.decoder(masked_features_with_cls, backward_indexes)
            x_recon = torch.clamp(x_recon, 0, 1)
        else:
            mask_probs_used = mask_probs
        
        # Upsample mask probabilities from patch resolution to pixel resolution
        num_patches = mask_probs_used.shape[1]
        num_patches_per_side = int(np.sqrt(num_patches))
        mask_spatial = mask_probs_used.view(-1, num_patches_per_side, num_patches_per_side)
        mask_spatial = mask_spatial.unsqueeze(1)  # Add channel dimension: (B, 1, H', W')
        
        image_size = pl_module.hparams.image_size
        mask_spatial = F.interpolate(
            mask_spatial,
            size=(image_size, image_size),
            mode='nearest'  # Use nearest neighbor to maintain sharp patch boundaries
        )  # (B, 1, H, W)
        
        # Expand to three channels for RGB
        mask_spatial_rgb = mask_spatial.expand(-1, 3, -1, -1)  # (B, 3, H, W)
        
        
        # Column 1: Original image (ground truth reference)
        original = val_img_samples
        
        # Column 2: Mask visualization (white for transmitted, black for masked)
        mask_vis = mask_spatial.expand(-1, 3, -1, -1)
        
        # Column 3: Composite image
        composite = mask_spatial_rgb * original + (1 - mask_spatial_rgb) * x_recon
        composite = torch.clamp(composite, 0, 1)
        
        # Concatenate all four columns horizontally to create the final visualization grid
        img = torch.cat([original, mask_vis, composite], dim=0)
        
        # Rearrange into a grid format for display
        img = rearrange(img, "(v h1 w1) c h w -> c (h1 h) (w1 v w)", w1=4, v=3)
        img = torch.clamp(img, 0, 1)
        
        mask_type = "Hard (Binary)" if self.use_hard_masks else "Soft (Continuous)"
        avg_transmitted = mask_probs_used.mean().item()
        
        caption = (
            f"Epoch {trainer.current_epoch} | {mask_type} Masks | "
            f"Transmission Rate: {avg_transmitted:.1%}\n"
            f"[Original | Mask (White=TX, Black=Masked) | Composite (Receiver)]"
        )
        
        log_name = "masking_viz_hard" if self.use_hard_masks else "masking_viz_soft"
        
        trainer.logger.experiment.log(
            {
                log_name: wandb.Image(img, caption=caption)
            }
        )