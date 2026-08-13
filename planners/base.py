from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from world.world import EnvironmentSnapshot


Point3D = Tuple[float, float, float]


@dataclass
class PlannerResult:
    success: bool
    path: List[Point3D]
    metadata: Dict[str, Any] = field(default_factory=dict)
    message: str = ""


class PathPlanner(ABC):
    """Единый интерфейс для A*, RRT, RRT*, PRM и других planner'ов."""

    def __init__(self, name: str | None = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def plan(
        self,
        start: Point3D,
        goal: Point3D,
        environment: "EnvironmentSnapshot",
    ) -> PlannerResult:
        raise NotImplementedError

    def reset(self) -> None:
        """Опциональный сброс внутреннего графа/дерева planner'а."""
        return None
