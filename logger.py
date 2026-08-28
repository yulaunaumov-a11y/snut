"""This module contains implementation of logger for ClearML."""
import datetime
import logging
from pathlib import Path

import torch

class Logger:
    """Implements logging to ClearML."""

    def __init__(self, config: Dict) -> None:
        """
        Initialize logger.

        Args:
            config: Project configuration object.
        """
        self.config = config
        self.job_time = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%d_%H:%M:%S")
        self.logger = logging.getLogger(__name__)
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.folder_name_to_save = Path(config.path.path_weights_to_save) / f"weights_{timestamp}"
        self.folder_name_to_save.mkdir(parents=True, exist_ok=True)

    def log_loss(self, epoch: int, train_loss: float, optimizer, metrics, text) -> None:
        """
        Log train and validation loss.

        Args:
            epoch: Epoch number;
            train_loss: Train loss;
        """
        self.logger.info(
            f"Epoch %d/%d: {text} metrics loss=%.6f accuracy=%.4f f1=%.4f learning_rate=%.6g",
            epoch,
            self.config.training.epochs,
            loss,
            metrics["accuracy"],
            metrics["f1"],
            optimizer.param_groups[0]["lr"],
        )
    
    def log_test(self, test_metrics):
        self.logger.info("Test metrics: %s", test_metrics)

    def log_model_weights(self, model_weights: Dict[str, torch.Tensor], epoch: int, name: str) -> None:
        """
        Save model's weights on the best epoch to the ClearML.

        Args:
            epoch: Number of epoch with best validation metric;
            model_weights: Model's state dictionary.
        """
        for file_path in self.folder_name_to_save.glob(f"{name}_*.pth"):
            file_path.unlink()

        file_name = f"{name}_{self.config.project.name}_epoch-{epoch}.pth"
        torch.save(model.state_dict(), self.folder_name_to_save / file_name)
