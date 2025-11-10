from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.loggers import Logger
from lightning.pytorch.trainer import Trainer
from lightning import LightningModule
import torch
import os
import wandb
import uuid

torch.set_float32_matmul_precision('high')

class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if isinstance(trainer.logger, Logger):
            config = self.parser.dump(self.config, skip_none=False)
            unique_id = str(uuid.uuid4())
            with open(f"run_config_{unique_id}.yaml", "w") as f:
                f.write(config)

            run = trainer.logger.experiment # type: ignore
            
            artifact = wandb.Artifact(name="config-file", type="config")
            artifact.add_file(f"run_config_{unique_id}.yaml")
            
            run.log_artifact(artifact)
            print(f"Uploaded run_config_{unique_id}.yaml to wandb!")
            os.remove(f"run_config_{unique_id}.yaml")

class TorchCompileCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--compile", action="store_true")
        parser.add_argument("--compile_kwargs", default={})

    def before_fit(self):
        if self.config["fit"]["compile"]:
            if hasattr(self.model, 'masking_model'):
                self.model.masking_model = torch.compile(
                    self.model.masking_model, 
                    **self.config["fit"]["compile_kwargs"]
                )

            elif hasattr(self.model, 'model'):
                self.model.model = torch.compile(
                    self.model.model, 
                    **self.config["fit"]["compile_kwargs"]
                )

        # set model.init_args.total_batch_size to the value set in the data field
        num_devices = self.config["fit"]["trainer"]["devices"]
        batch_size = self.config["fit"]["data"]["init_args"]["batch_size"]
        accumulate_grad_batches = self.config["fit"]["trainer"]["accumulate_grad_batches"]

        total_batch_size = batch_size * num_devices * accumulate_grad_batches
        self.config["fit"]["model"]["init_args"]["total_batch_size"] = total_batch_size

def main():
    cli = TorchCompileCLI(save_config_callback=LoggerSaveConfigCallback, save_config_kwargs={"save_to_log_dir": False})

if __name__ == "__main__":
    main()