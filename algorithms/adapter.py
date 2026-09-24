from __future__ import annotations

import math
import time
from typing import Any, Optional, Sequence

from algorithms.base import AlgorithmType, NavigationAlgorithm
from algorithms.policies.base import PolicyController
from planners.base import PathPlanner, PlannerResult
from world.world import SimulationWorld


class AlgorithmAdapter:
    """
    Единая точка подключения алгоритмов к стенду.

    Поддерживает:

    1. Классические алгоритмы:
       A*, RRT, RRT*, PRM ...

    2. Policy-based алгоритмы:
       PPO, SAC, TD3 ...
    """

    def __init__(
        self,
        algorithm: NavigationAlgorithm,
    ):
        self.algorithm = algorithm

    @property
    def algorithm_type(self) -> AlgorithmType:
        return self.algorithm.algorithm_type

    def set_algorithm(
        self,
        algorithm: NavigationAlgorithm,
    ) -> None:
        """
        Позволяет заменить алгоритм во время работы стенда.
        """

        self.algorithm = algorithm

    def reset(self) -> None:
        """
        Сброс подключённого алгоритма.
        """

        self.algorithm.reset()

    def plan(
        self,
        world: SimulationWorld,
        start: Optional[Sequence[float]] = None,
        goal: Optional[Sequence[float]] = None,
    ) -> PlannerResult:
        """
        Запуск классического алгоритма поиска пути.
        """

        if not isinstance(self.algorithm, PathPlanner):
            raise TypeError(
                f"Алгоритм '{self.algorithm.name}' "
                f"не является PathPlanner"
            )

        snapshot = world.snapshot()

        planner_start = (
            tuple(start)
            if start is not None
            else snapshot.start
        )

        planner_goal = (
            tuple(goal)
            if goal is not None
            else snapshot.goal
        )

        self._validate_point(
            planner_start,
            snapshot.bounds,
            "start",
        )

        self._validate_point(
            planner_goal,
            snapshot.bounds,
            "goal",
        )

        started = time.perf_counter()

        result = self.algorithm.plan(
            start=planner_start,
            goal=planner_goal,
            environment=snapshot,
        )

        elapsed = time.perf_counter() - started

        if not isinstance(result, PlannerResult):
            raise TypeError(
                f"Алгоритм '{self.algorithm.name}' "
                f"должен возвращать PlannerResult"
            )

        result.metadata.setdefault(
            "algorithm",
            self.algorithm.name,
        )

        result.metadata.setdefault(
            "algorithm_type",
            self.algorithm_type.value,
        )

        result.metadata.setdefault(
            "planning_time",
            elapsed,
        )

        if result.success and result.path:
            result.metadata.setdefault(
                "path_length",
                self._calculate_path_length(
                    result.path
                ),
            )

        return result

    def act(
        self,
        observation: Any,
        world: SimulationWorld | None = None,
    ) -> Any:
        """
        Получить action от ML/policy алгоритма.
        """

        if not isinstance(
            self.algorithm,
            PolicyController,
        ):
            raise TypeError(
                f"Алгоритм '{self.algorithm.name}' "
                f"не является PolicyController"
            )

        snapshot = (
            world.snapshot()
            if world is not None
            else None
        )

        return self.algorithm.act(
            observation=observation,
            environment=snapshot,
        )

    @staticmethod
    def _validate_point(
        point,
        bounds,
        name: str,
    ) -> None:
        if len(point) != 3:
            raise ValueError(
                f"{name} должен содержать 3 координаты"
            )

        minimum, maximum = bounds

        for i in range(3):
            if not minimum[i] <= point[i] <= maximum[i]:
                raise ValueError(
                    f"{name}={point} находится "
                    f"за пределами арены {bounds}"
                )

    @staticmethod
    def _calculate_path_length(
        path,
    ) -> float:
        total = 0.0

        for current, next_point in zip(
            path[:-1],
            path[1:],
        ):
            total += math.dist(
                current,
                next_point,
            )

        return total