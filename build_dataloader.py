import albumentations
import pandas as pd
import torch
import numpy as np 
import random
import cv2

from tqdm import tqdm
from pathlib import Path, PosixPath
from torch.utils.data import DataLoader
from video_dataset import VideoDataset
from utils import read_config
from typing import Dict, List, Tuple

def build_dataloaders(
    config: Dict, 
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    dataset = VideoDataset(
        config.path.csv_path,
        sampling_strategy=config.model.sampling_strategy,
        video_loader=DecordVideo,
        num_frames=config.model.num_frames,
    )
    split_lengths = [
        int(len(dataset) * 0.9),
        int(len(dataset) * 0.1),
    ]
    split_lengths.append(len(dataset) - sum(split_lengths))

    train_dataset, validation_dataset, test_dataset = random_split(
        dataset,
        split_lengths,
        generator=torch.Generator(),
    )
    
    train_data_loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size_train,
        shuffle=True,
        num_workers=config.training.num_workers,
    )
    val_data_loader = DataLoader(
        validation_dataset,
        batch_size=config.training.batch_size_train,
        shuffle=True,
        num_workers=config.training.num_workers,
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size_test,
        shuffle=True,
        num_workers=config.training.num_workers,
    )

    return train_data_loader, test_data_loader, val_data_loader