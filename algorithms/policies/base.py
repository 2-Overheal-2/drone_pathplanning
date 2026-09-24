from __future__ import annotations

from abc import abstractmethod
from typing import Any, TYPE_CHECKING

from algorithms.base import AlgorithmType, NavigationAlgorithm


if TYPE_CHECKING:
    from world.world import EnvironmentSnapshot


class PolicyController(NavigationAlgorithm):
    """
    Базовый интерфейс policy-based алгоритмов.

    Например:
    PPO
    SAC
    TD3
    пользовательская нейросеть
    """

    algorithm_type = AlgorithmType.POLICY

    @abstractmethod
    def act(
        self,
        observation: Any,
        environment: "EnvironmentSnapshot | None" = None,
    ) -> Any:
        """
        Получить управляющее действие по текущему observation.
        """
        raise NotImplementedError