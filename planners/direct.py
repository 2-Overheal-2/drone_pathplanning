from __future__ import annotations

from planners.base import PathPlanner, PlannerResult, Point3D
from world.world import EnvironmentSnapshot


class DirectPlanner(PathPlanner):
    """Тестовый planner: возвращает прямой отрезок, препятствия не учитывает."""

    def plan(
        self,
        start: Point3D,
        goal: Point3D,
        environment: EnvironmentSnapshot,
    ) -> PlannerResult:
        return PlannerResult(
            success=True,
            path=[start, goal],
            message="Direct path generated; obstacle avoidance is not implemented.",
        )
