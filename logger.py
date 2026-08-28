"""This module contains implementation of logger for ClearML."""
from pathlib import Path
import datetime
import cv2
import addict
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from typing import Dict
from torch.utils.data import DataLoader
from utils import read_config
from sklearn.metrics import confusion_matrix

import io
from PIL import Image

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
        self.logger = logging.getLogger()
        
        self.folder_name_to_save = Path(self.config.path.path_weights_to_save, f"weights_{datetime.datetime.now().isoformat()}")
        self.folder_name_to_save.mkdir(exist_ok=True)

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
        time = datetime.datetime.now()

        matching_files = list(self.folder_name_to_save.glob(f"*{name}*"))

        for file_path in matching_files:
            if file_path.is_file():
                file_path.unlink()

        file_name = f"{name}_{self.config.project.name}_{time}_epoch:{epoch}.pth"
        weights_path = Path(self.folder_name_to_save.as_posix(), file_name)
        torch.save(model_weights.state_dict(), weights_path.as_posix())
        self.task.upload_artifact("model_weights.pth", artifact_object=weights_path.as_posix())
