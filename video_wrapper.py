from abc import ABC, abstractmethod

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


class DecordVideo(IVideoWrapper): ...

class OpenCVVideo(IVideoWrapper): ...
