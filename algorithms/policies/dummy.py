from __future__ import annotations

import numpy as np

from algorithms.policies.base import PolicyController
from world.world import EnvironmentSnapshot


class DummyPolicyController(PolicyController):
    """
    Простая тестовая policy.

    Не является навигационным алгоритмом.
    Используется только для проверки архитектуры.
    """

    def __init__(self):
        super().__init__(
            name="DummyPolicy"
        )

    def reset(self) -> None:
        pass

    def act(
        self,
        observation,
        environment: EnvironmentSnapshot | None = None,
    ):
        return np.zeros(
            (1, 4),
            dtype=np.float32,
        )