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

from logger import Logger
from video_dataset import VideoDataset
from video_model import vit_base_patch16_224
from video_wrapper import DecordVideo
from utils import read_config

def get_args():
    parser = argparse.ArgumentParser(
        "Run Video Model training.",
        add_help=True)
    parser.add_argument(
        "--path_to_annotation_file", 
        type=str, 
        required=True, 
        help="Path to the CSV file containing annotations."
        )
    parser.add_argument(
        "--num_classes",
        default=10,
        type=int,
        help="Number of action classes. Defaults to 10."
    )
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="Device for training model. Defaults to \"cpu\"."
    )
    parser.add_argument(
        '--batch_size',
        default=8,
        type=int,
        help="Batch size. Defaults to 8."
    )
    parser.add_argument(
        "--num_epochs",
        default=10,
        type=int,
        help="Number of epochs. Defaults to 10."
    )
    parser.add_argument(
        "--lr",
        default=1e-3,
        type=float,
        help="Initial learning rate. Defaults to 1e-3."
    )
    parser.add_argument(
        "--sampling_strategy",
        default="fixed",
        type=str,
        help="Frame sampling strategy from one video clip. Defaults to \"fixed\"."
    )
    parser.add_argument(
        "--path_to_save",
        type=str,
        help="Path to save trained model state_dict in .pth format."
    )
    parser.add_argument(
        "--num_frames", 
        default=16, 
        type=int, 
        help="Frames per video clip."
    )
    parser.add_argument(
        "--num_workers", 
        default=0, 
        type=int, 
        help="DataLoader worker count."
    )
    parser.add_argument(
        "--path_to_save", 
        type=str, 
        help="Output .pth model state_dict path."
    )
    
    return parser.parse_known_args()


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
    dataset = VideoDataset(
        config.path.csv_path,
        sampling_strategy=config.model.sampling_strategy,
        video_loader=DecordVideo,
        num_frames=config.model.num_frames,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size_train,
        shuffle=True,
        num_workers=config.training.num_workers,
    )
    print("Initializing logging.")
    logger = Logger(config)

    print("Start the learning process.")
    for epoch in tqdm(range(config.training.epochs), total=config.training.epochs, desc="Training"):
        loss = train_one_epoch(model, criterion, data_loader, optimizer, scheduler, device)
        logger.log_loss(epoch=epoch, train_loss=loss, optimizer=optimizer)

    logger.log_model_weights(model_weights=model, epoch=epoch, name = "last")
   print("The learning process is complete.")