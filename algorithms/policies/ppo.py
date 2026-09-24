from __future__ import annotations

from typing import Any

from stable_baselines3 import PPO

from algorithms.policies.base import PolicyController
from world.world import EnvironmentSnapshot


class PPOPolicyController(PolicyController):
    """
    Адаптер обученной Stable-Baselines3 PPO модели.

    Стенд не должен знать детали Stable-Baselines3.
    Для стенда это просто PolicyController:
        observation -> action
    """

    def __init__(
        self,
        model_path: str,
        deterministic: bool = True,
        device: str = "cpu",
    ):
        super().__init__(name="PPO")

        self.model_path = model_path
        self.deterministic = deterministic
        self.device = device

        self.model = PPO.load(
            model_path,
            device=device,
        )

    def reset(self) -> None:
        """
        Для обычной MLP PPO policy отдельного
        recurrent state нет, поэтому ничего делать не нужно.
        """
        pass

    def act(
        self,
        observation: Any,
        environment: EnvironmentSnapshot | None = None,
    ) -> Any:
        action, _ = self.model.predict(
            observation,
            deterministic=self.deterministic,
        )

        return action