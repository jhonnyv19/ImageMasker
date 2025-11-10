import lightning as L
import torch
import math
import wandb
from einops import rearrange

from ..models.mae import MAE_ViT


class LitMAE(L.LightningModule):
    def __init__(
        self,
        lr: float = 1.5e-4,
        weight_decay: float = 0.05,
        mask_ratio: float = 0.75,
        warmup_epochs: int = 200,
        total_batch_size: int = 4096,
        image_size: int = 32,
        patch_size: int = 2,
        enc_emb_dim: int = 192,
        dec_emb_dim: int = 192,
        encoder_layer: int = 12,
        encoder_head: int = 3,
        decoder_layer: int = 4,
        decoder_head: int = 3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = MAE_ViT(
            image_size=self.hparams.image_size,
            patch_size=self.hparams.patch_size,
            enc_emb_dim=self.hparams.enc_emb_dim,
            dec_emb_dim=self.hparams.dec_emb_dim,
            encoder_layer=self.hparams.encoder_layer,
            encoder_head=self.hparams.encoder_head,
            decoder_layer=self.hparams.decoder_layer,
            decoder_head=self.hparams.decoder_head,   
            mask_ratio=self.hparams.mask_ratio,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, _ = batch
        predicted_img, mask = self(img)
        loss = torch.mean((predicted_img - img) ** 2 * mask) / self.hparams.mask_ratio
        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, _ = batch
        predicted_img, mask = self(img)
        loss = torch.mean((predicted_img - img) ** 2 * mask) / self.hparams.mask_ratio
        self.log("val/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        
        # Cache first batch for visualization
        if batch_idx == 0:
            self.last_val_outputs = {
                'img': img[:16].detach(),
                'predicted_img': predicted_img[:16].detach(),
                'mask': mask[:16].detach()
            }
        
        return loss

    def on_validation_epoch_end(self):
        if not hasattr(self, 'last_val_outputs'):
            return
        
        # Use cached outputs
        val_img_samples = self.last_val_outputs['img']
        predicted_img = self.last_val_outputs['predicted_img']
        mask = self.last_val_outputs['mask']
        
        predicted_img = predicted_img * mask + val_img_samples * (1 - mask)
        img = torch.cat(
            [val_img_samples * (1 - mask), predicted_img, val_img_samples], dim=0
        )
        img = rearrange(img, "(v h1 w1) c h w -> c (h1 h) (w1 v w)", w1=2, v=3)
        img = torch.clamp(img, 0, 1)
        self.logger.experiment.log(
            {
                "reconstructions": wandb.Image(
                    img, caption=f"Epoch {self.current_epoch}"
                )
            }
        )

    def configure_optimizers(self):
        # Scale learning rate for 256 size batch proportional to 2048 size batch
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr * self.hparams.total_batch_size / 256,
            betas=(0.9, 0.95),
            weight_decay=self.hparams.weight_decay,
        )

        # min(epoch / warmup_epochs, 0.5 * (cos(epoch / max_epochs * pi) + 1))
        def lr_func(epoch):
            return min(
                (epoch + 1) / (self.hparams.warmup_epochs + 1e-8),
                0.5 * (math.cos(epoch / self.trainer.max_epochs * math.pi) + 1),
            )

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optim, lr_lambda=lr_func
        )
        return [optim], [lr_scheduler]
