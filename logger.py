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

    def log_loss(self, epoch: int, train_loss: float, optimizer) -> None:
        """
        Log train and validation loss.

        Args:
            epoch: Epoch number;
            train_loss: Train loss;
        """
        self.logger.info(
            "Epoch %d/%d: loss=%.6f learning_rate=%.6g",
            epoch,
            self.config.training.epochs,
            train_loss,
            optimizer.param_groups[0]["lr"],
        )


    def log_train_and_val_metrics(self, epoch: int, metrics_dict: Dict, title:str = "Metrics") -> None:
        """
        Log train and validation metrics.

        Args:
            epoch: Epoch number;
            metrics_dict_val: Results of evaluation encapsulated in separate object.
        """

        self.logger.report_scalar(
            title=title,
            series="Jaccard",
            iteration=epoch,
            value=metrics_dict["Jaccard"],
        )
        self.logger.report_scalar(
            title=title,
            series="F1",
            iteration=epoch,
            value=metrics_dict["F1"],
        )
        self.logger.report_scalar(
            title=title,
            series="Recall",
            iteration=epoch,
            value=metrics_dict["Recall"],
        )
        self.logger.report_scalar(
            title=title,
            series="Precision",
            iteration=epoch,
            value=metrics_dict["Precision"],
        )
        self.logger.report_scalar(
            title=title,
            series="Acc",
            iteration=epoch,
            value=metrics_dict["Acc"],
        )


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
