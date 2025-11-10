import lightning as L
import torch
import torch.nn.functional as F
import wandb
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import math

from ..models.mae import MAE_ViT
from ..models.mask_predictor import ImageTransformerMaskPredictor
from ..models.masking_model import MaskingModel as MaskingModelBase
class LitMaskingModel(L.LightningModule):
    def __init__(
        self,
        lr: float = 1.5e-4,
        weight_decay: float = 0.01,
        mean_reg: float = 3e-3,
        entropy_reg: float = 1e-3,
        epsilon: float = 1e-6,
        image_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 192,
        num_layers: int = 2,
        num_heads: int = 4,
        mae_checkpoint_path: str = "weights/mae-cifar10.ckp",
        log_interval: int = 50,
        total_batch_size: int = 1024,
        use_gumbel_softmax: bool = False,
        tau_config: dict = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Load pretrained MAE
        checkpoint = torch.load(mae_checkpoint_path)
        if image_size == 96:
            mae_kwargs = {
                "image_size": image_size,
                "patch_size": patch_size,
                "enc_emb_dim": 516,
                "encoder_layer": 12,
                "encoder_head": 12,
                "dec_emb_dim": 516,
                "decoder_layer": 8,
                "decoder_head": 12,
            }

        elif image_size == 32:
            mae_kwargs = {
                "image_size": image_size,
                "patch_size": patch_size,
                "enc_emb_dim": 192,
                "encoder_layer": 12,
                "encoder_head": 3,
                "dec_emb_dim": 192,
                "decoder_layer": 4,
                "decoder_head": 3,
            }

        mae = MAE_ViT(**mae_kwargs)
        state_dict = checkpoint["state_dict"]
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        mae.load_state_dict(state_dict)

        num_patches = (image_size // patch_size) ** 2

        # Create transformer-based mask predictor
        mask_predictor = ImageTransformerMaskPredictor(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_patches=num_patches,
            num_layers=num_layers,
            num_heads=num_heads,
        )

        self.masking_model = MaskingModelBase(mae, mask_predictor, use_gumbel_softmax=use_gumbel_softmax)
        self.use_gumbel_softmax = use_gumbel_softmax
        self.tau_config = tau_config

        # cast all fields to int
        self.tau_config = {
            "start_value": float(self.tau_config["start_value"]),
            "final_value": float(self.tau_config["final_value"]),
            "start_step": int(self.tau_config["start_step"]),
            "final_step": int(self.tau_config["final_step"]),
        }
        self.tau = 0.5


    def _cosine_anneal(
        self,
        step: int,
        start_value: float,
        final_value: float,
        start_step: int,
        final_step: int,
    ) -> float:
        """
        Cosine annealing scheduling function, copied from SlateHDCTrainer.

        Args:
            step (int): Current training step.
            start_value (float): Initial value.
            final_value (float): Final value after annealing.
            start_step (int): Step at which to start annealing.
            final_step (int): Step at which to end annealing.

        Returns:
            float: The annealed value at the current step.
        """
        if step < start_step:
            return start_value
        if step >= final_step:
            return final_value

        a = 0.5 * (start_value - final_value)
        b = 0.5 * (start_value + final_value)
        progress = (step - start_step) / (final_step - start_step)
        return a * math.cos(math.pi * progress) + b

    def on_train_batch_start(self, batch, batch_idx):
        if self.tau_config and self.trainer is not None:
            self.tau = self._cosine_anneal(
                self.trainer.global_step, **self.tau_config
            )
        else:
            self.tau = 0.5
        self.log("train/tau", self.tau, on_step=True, on_epoch=True, sync_dist=False)

    def forward(self, x):
        return self.masking_model(x, tau=self.tau)

    def _calculate_loss(self, batch):
        x, _ = batch
        x_recon, mask_probs = self(x)

        # Upsample mask probabilities to image resolution
        num_patches_per_side = int(np.sqrt(mask_probs.shape[1]))
        mask_probs_spatial = mask_probs.view(-1, num_patches_per_side, num_patches_per_side)
        mask_probs_spatial = mask_probs_spatial.unsqueeze(1)  # (B, 1, H', W')
        
        # Upsample to full image resolution
        mask_probs_spatial = F.interpolate(
            mask_probs_spatial, 
            size=(self.hparams.image_size, self.hparams.image_size), 
            mode='nearest'
        )  # (B, 1, H, W)
        
        # Expand to three channels
        mask_probs_spatial = mask_probs_spatial.expand(-1, 3, -1, -1)  # (B, 3, H, W)
        
        # MSE Loss
        x_composite = mask_probs_spatial * x + (1 - mask_probs_spatial) * x_recon
        x_composite = torch.clamp(x_composite, 0, 1)
        mse_loss = F.mse_loss(x_composite, x)
        
        # Regularization terms
        mask_mean = mask_probs.mean()  # Encourage sparsity (fewer transmitted patches)
        entropy = -(
            mask_probs * torch.log(mask_probs + self.hparams.epsilon)
            + (1 - mask_probs) * torch.log(1 - mask_probs + self.hparams.epsilon)
        ).mean()  # Encourage confident decisions

        # Combined loss
        total_loss = (
            mse_loss
            + self.hparams.mean_reg * mask_mean**2
            + self.hparams.entropy_reg * entropy
        )

        return total_loss, mse_loss, mask_mean, entropy, mask_probs

    def training_step(self, batch, batch_idx):
        total_loss, mse, mask_mean, entropy, mask_probs = self._calculate_loss(batch)
        
        self.log("train/loss", total_loss, on_step=True, on_epoch=True, sync_dist=False)
        self.log("train/mse", mse, on_step=True, on_epoch=True, sync_dist=False)
        self.log("train/mask_mean", mask_mean, on_step=True, on_epoch=True, sync_dist=False)
        self.log("train/entropy", entropy, on_step=True, on_epoch=True, sync_dist=False)

        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, mse, mask_mean, entropy, mask_probs = self._calculate_loss(batch)
        self.log("val/loss", total_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/mse", mse, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/mask_mean", mask_mean, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/entropy", entropy, on_step=False, on_epoch=True, sync_dist=True)
        
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.masking_model.mask_predictor.parameters(),
            lr=self.hparams.lr * self.hparams.total_batch_size / 256,
            weight_decay=self.hparams.weight_decay,
        )

        def lr_lambda(current_step):
            num_epochs = self.trainer.max_epochs
            if not self.trainer.datamodule:
                return 1.0

            total_steps = num_epochs * len(self.trainer.datamodule.train_dataloader())
            warmup_steps = int(0.1 * total_steps)
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            else:
                return max(
                    0.0,
                    float(total_steps - current_step)
                    / float(max(1, total_steps - warmup_steps)),
                )

        scheduler = LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
