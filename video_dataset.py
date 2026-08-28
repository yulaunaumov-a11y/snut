import csv
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import Dataset

from video_wrapper import IVideoWrapper


class VideoDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        path_to_annotation_file: str | Path,
        sampling_strategy: Literal["fixed", "random"],
        video_loader: type[IVideoWrapper],
        num_frames: int = 16,
    ) -> None:

        annotation_path = Path(path_to_annotation_file)

        with annotation_path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            expected_columns = {"path_to_video", "class_label"}
            self.annotations = [
                (row["path_to_video"], int(row["class_label"])) for row in reader
            ]

        self.sampling_strategy = sampling_strategy
        self.video_loader = video_loader
        self.num_frames = num_frames
    def __len__(self) -> int:
        return len(self.annotations)

    def _sample_indices(self, video_length: int) -> np.ndarray:

        if self.sampling_strategy == "fixed":
            return np.linspace(0, video_length - 1, self.num_frames, dtype=np.int64)

        boundaries = np.linspace(0, video_length, self.num_frames + 1, dtype=np.int64)
        starts, ends = boundaries[:-1], boundaries[1:]

        ends = np.maximum(ends, starts + 1)
        starts = np.minimum(starts, video_length - 1)
        ends = np.minimum(ends, video_length)
        return np.asarray(
            [np.random.randint(start, end) for start, end in zip(starts, ends)],
            dtype=np.int64,
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path_to_video, label = self.annotations[index]
        video = self.video_loader(path_to_video)
        frames = video.get_batch(self._sample_indices(video.get_length()).tolist())

        clip = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)
        clip = clip.to(dtype=torch.float32).div_(255.0)
        clip = functional.interpolate(clip, size=(224, 224), mode="bilinear", align_corners=False)
        return clip, torch.tensor(label, dtype=torch.long)

