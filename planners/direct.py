from __future__ import annotations

from planners.base import (
    PathPlanner,
    PlannerResult,
    Point3D,
)

from world.world import EnvironmentSnapshot


class DirectPlanner(PathPlanner):
    """
    Простейший тестовый планировщик.

    Возвращает:
        start -> goal

    Препятствия намеренно не учитываются.
    """

    def __init__(self):
        super().__init__(
            name="DirectPlanner"
        )

    def plan(
        self,
        start: Point3D,
        goal: Point3D,
        environment: EnvironmentSnapshot,
    ) -> PlannerResult:

        return PlannerResult(
            success=True,

            path=[
                start,
                goal,
            ],

            message=(
                "Direct path generated. "
                "Obstacle avoidance is disabled."
            ),
        )