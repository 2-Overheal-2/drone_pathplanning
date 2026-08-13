from __future__ import annotations

import math
import time
from typing import Optional, Sequence

from planners.base import PathPlanner, PlannerResult, Point3D
from world.world import SimulationWorld


class PlannerAdapter:
    """Связка SimulationWorld -> EnvironmentSnapshot -> PathPlanner."""

    def __init__(self, planner: PathPlanner):
        self.planner = planner

    def set_planner(self, planner: PathPlanner) -> None:
        self.planner = planner

    def run(
        self,
        world: SimulationWorld,
        start: Optional[Sequence[float]] = None,
        goal: Optional[Sequence[float]] = None,
    ) -> PlannerResult:
        snapshot = world.snapshot()

        planner_start = tuple(start) if start is not None else tuple(snapshot.start)
        planner_goal = tuple(goal) if goal is not None else tuple(snapshot.goal)

        self._validate_point(planner_start, snapshot.bounds, "start")
        self._validate_point(planner_goal, snapshot.bounds, "goal")

        started_at = time.perf_counter()
        result = self.planner.plan(
            start=planner_start,
            goal=planner_goal,
            environment=snapshot,
        )
        elapsed = time.perf_counter() - started_at

        if not isinstance(result, PlannerResult):
            raise TypeError(
                f"Planner '{self.planner.name}' должен возвращать PlannerResult, "
                f"а вернул {type(result).__name__}."
            )

        result.metadata.setdefault("planner", self.planner.name)
        result.metadata.setdefault("planning_time", elapsed)

        if result.success and result.path:
            result.metadata.setdefault("path_length", self._path_length(result.path))

        return result

    @staticmethod
    def _validate_point(point, bounds, name: str) -> None:
        if len(point) != 3:
            raise ValueError(f"{name} должен содержать 3 координаты.")

        minimum, maximum = bounds
        if not all(minimum[i] <= point[i] <= maximum[i] for i in range(3)):
            raise ValueError(f"{name}={point} находится вне bounds={bounds}.")

    @staticmethod
    def _path_length(path) -> float:
        total = 0.0
        for current, nxt in zip(path[:-1], path[1:]):
            total += math.dist(current, nxt)
        return total
