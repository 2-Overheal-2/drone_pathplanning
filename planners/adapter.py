from __future__ import annotations

from typing import Optional, Sequence

from algorithms.adapter import AlgorithmAdapter

from planners.base import (
    PathPlanner,
    PlannerResult,
)

from world.world import SimulationWorld


class PlannerAdapter:
    """
    Специализированный адаптер для PathPlanner.

    Оставлен для совместимости со старым кодом.

    В новом коде рекомендуется использовать:
        AlgorithmAdapter
    """

    def __init__(
        self,
        planner: PathPlanner,
    ):
        self._adapter = AlgorithmAdapter(
            planner
        )

    @property
    def planner(self) -> PathPlanner:
        return self._adapter.algorithm

    def set_planner(
        self,
        planner: PathPlanner,
    ) -> None:
        """
        Заменить planner без создания нового адаптера.
        """

        self._adapter.set_algorithm(
            planner
        )

    def reset(self) -> None:
        self._adapter.reset()

    def run(
        self,
        world: SimulationWorld,
        start: Optional[Sequence[float]] = None,
        goal: Optional[Sequence[float]] = None,
    ) -> PlannerResult:
        """
        Запустить planner на текущем состоянии мира.
        """

        return self._adapter.plan(
            world=world,
            start=start,
            goal=goal,
        )