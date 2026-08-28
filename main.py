"""
    Пример запуска скрипта:
        $ python3 main.py /PATH/TO/ANNOTATION/FILE.csv
"""

import argparse
import torch

from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from build_dataloader import build_dataloaders
from evaluate import evaluate
from logger import Logger
from video_dataset import VideoDataset
from video_model import vit_base_patch16_224
from video_wrapper import DecordVideo
from utils import read_config

def train_one_epoch(model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    scheduler,
                    device: torch.device
                    ):
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for step, (inputs, labels) in enumerate(data_loader, start=1):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True
                           
        y_pred = model(inputs)
                           
        loss = criterion(y_pred, labels)
        loss.backward()
        optimizer.step()
        batch_size = labels.size(0)
        total_loss += loss.detach().item() * batch_size
        total_samples += batch_size

   scheduler.step()
   return total_loss / total_samples
                           
if __name__ == "__main__":

    config = read_config("./config.yml")
                           
    print("Declare a device.")
    device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
    print(f"Device use: {device}")
                           
    print("Initializing the model.")
    model = vit_base_patch16_224(num_classes=config.task.num_classes).to(device)

    print("Initializing criterion, optimizer.")
    optimizer = object_from_dict(config.optimizer, params = model.parameters())
    criterion = object_from_dict(config.criterion)
    scheduler = object_from_dict(config.scheduler)

    print("Build a dataloaders.")
    train_data_loader, test_data_loader, val_data_loader = build_dataloaders(config)
    print("Initializing logging.")
    logger = Logger(config)

    print("Start the learning process.")
    old_metrics = {
        "accuracy": -1,
        "precision": -1,
        "recall": -1,
        "f1": -1,
    }
    for epoch in tqdm(range(config.training.epochs), total=config.training.epochs, desc="Training"):
        loss = train_one_epoch(model, criterion, data_loader, optimizer, scheduler, device)

        validation_metrics = evaluate(
            model,
            val_data_loader,
            device=device,
            batch_size=config.training.batch_size_train,
            num_workers=config.training.num_workers,
        )
        train_metrics = evaluate(
            model,
            train_data_loader,
            device=device,
            batch_size=config.training.batch_size_train,
            num_workers=config.training.num_workers,
        )
        if old_metrics["f1"] < validation_metrics["f1"]:
            logger.log_model_weights(model_weights=model, epoch=epoch, name = "best_model")
            old_metrics = validation_metrics

        logger.log_loss(
           epoch=epoch, 
           train_loss=loss, 
           optimizer=optimizer, 
           metrics=train_metrics, 
           text="train"
        )

        logger.log_loss(
           epoch=epoch, 
           train_loss=loss, 
           optimizer=optimizer, 
           metrics=validation_metrics, 
           text="validate"
        )

    test_metrics = evaluate(
        model,
        test_data_loader,
        device=device,
            batch_size=config.training.batch_size_test,
            num_workers=config.training.num_workers,
    )
    logger.log_test(test_metrics)
    logger.log_model_weights(model_weights=model, epoch=epoch, name = "last")
    print("The learning process is complete.")