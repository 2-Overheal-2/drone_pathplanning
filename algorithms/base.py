from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum


class AlgorithmType(str, Enum):
    """
    Тип подключённого навигационного алгоритма.
    """

    PATH_PLANNER = "path_planner"
    POLICY = "policy"


class NavigationAlgorithm(ABC):
    """
    Общий базовый класс для любого навигационного алгоритма.

    От него наследуются:
    - PathPlanner
    - PolicyController
    """

    algorithm_type: AlgorithmType

    def __init__(
        self,
        name: str | None = None,
    ):
        self.name = (
            name
            if name is not None
            else self.__class__.__name__
        )

    @abstractmethod
    def reset(self) -> None:
        """
        Сброс внутреннего состояния алгоритма
        перед новым эпизодом или запуском.
        """

        raise NotImplementedError