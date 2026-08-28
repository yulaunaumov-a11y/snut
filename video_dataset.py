from pathlib import Path
from typing import Literal

import torch

from video_wrapper import IVideoWrapper


class VideoDataset:
    def __init__(self,
                 path_to_annotation_file: str | Path,
                 sampling_strategy: Literal["fixed", "random"],
                 video_loader: type[IVideoWrapper],
                 num_frames: int = 16
                 ):
        super().__init__()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError