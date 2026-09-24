from __future__ import annotations

from typing import Optional, Sequence

from algorithms.adapter import AlgorithmAdapter
from algorithms.base import AlgorithmType

from planners.base import PlannerResult


class SimulationRunner:
    """
    Связывает среду и подключённый алгоритм.

    Для policy:
        observation
        -> algorithm
        -> action
        -> environment.step()

    Для planner:
        world.snapshot()
        -> algorithm
        -> path
    """

    def __init__(
        self,
        env,
        algorithm_adapter: AlgorithmAdapter,
    ):
        self.env = env

        self.algorithm_adapter = (
            algorithm_adapter
        )

        self.observation = None
        self.info = None

    @property
    def world(self):
        """
        Получить SimulationWorld из среды.
        """

        world = getattr(
            self.env,
            "world",
            None,
        )

        if world is None:
            raise RuntimeError(
                "Environment не содержит "
                "активный SimulationWorld"
            )

        return world

    def reset(
        self,
        seed: int | None = None,
    ):
        """
        Начать новый эпизод.
        """

        self.algorithm_adapter.reset()

        self.observation, self.info = (
            self.env.reset(
                seed=seed
            )
        )

        return (
            self.observation,
            self.info,
        )

    def policy_step(self):
        """
        Выполнить один шаг policy-based алгоритма.

        observation
            ↓
        PPO / другая policy
            ↓
        action
            ↓
        env.step(action)
        """

        if (
            self.algorithm_adapter.algorithm_type
            != AlgorithmType.POLICY
        ):
            raise TypeError(
                "Подключённый алгоритм "
                "не является PolicyController"
            )

        if self.observation is None:
            raise RuntimeError(
                "Перед policy_step() "
                "необходимо вызвать reset()"
            )

        action = self.algorithm_adapter.act(
            observation=self.observation,
            world=self.world,
        )

        result = self.env.step(
            action
        )

        self.observation = result[0]

        return result

    def plan_path(
        self,
        start: Optional[Sequence[float]] = None,
        goal: Optional[Sequence[float]] = None,
    ) -> PlannerResult:
        """
        Запустить классический path planner.
        """

        if (
            self.algorithm_adapter.algorithm_type
            != AlgorithmType.PATH_PLANNER
        ):
            raise TypeError(
                "Подключённый алгоритм "
                "не является PathPlanner"
            )

        return self.algorithm_adapter.plan(
            world=self.world,
            start=start,
            goal=goal,
        )