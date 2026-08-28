from abc import ABC, abstractmethod
import importlib
from pathlib import Path
from typing import Sequence

import numpy as np


class IVideoWrapper(ABC):
    @abstractmethod
    def get_frame(self, index: int) -> np.ndarray:
        """
        Возвращает кадр из видео по индексу
        """
        raise NotImplementedError

    @abstractmethod
    def get_batch(self, indices: list[int]) -> np.ndarray:
        """
        Возвращает набор кадров по индексам
        """
        raise NotImplementedError

    @abstractmethod
    def get_length(self) -> int:
        """
        Возвращает длительность видео в кадрах
        """
        raise NotImplementedError


class DecordVideo(IVideoWrapper):
    def __init__(self, path_to_video: str | Path) -> None:
        
        self.path = str(path_to_video)
        decord = importlib.import_module("decord")
        self._reader = decord.VideoReader(self.path, ctx=decord.cpu(0))

    def get_length(self) -> int:
        return len(self._reader)

    def get_frame(self, index: int) -> np.ndarray:
        return self._reader[index].asnumpy()

    def get_batch(self, indices: Sequence[int]) -> np.ndarray:
        
        indices_array = np.asarray(indices, dtype=np.int64)

        return self._reader.get_batch(indices_array).asnumpy()


class OpenCVVideo(IVideoWrapper):
    def __init__(self, path_to_video: str | Path) -> None:
        
        cv2 = importlib.import_module("cv2")
        self._cv2 = cv2
        self.path = str(path_to_video)
        self._capture = cv2.VideoCapture(self.path)
        self._length = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_length(self) -> int:
        return self._length

    def get_frame(self, index: int) -> np.ndarray:

        self._capture.set(self._cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = self._capture.read()

        return self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)

    def get_batch(self, indices: Sequence[int]) -> np.ndarray:
        return np.stack([self.get_frame(int(index)) for index in indices])

    def __del__(self) -> None:
        capture = getattr(self, "_capture", None)
        if capture is not None:
            capture.release()
