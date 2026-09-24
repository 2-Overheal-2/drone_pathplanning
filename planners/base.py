from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from algorithms.base import AlgorithmType, NavigationAlgorithm


if TYPE_CHECKING:
    from world.world import EnvironmentSnapshot


Point3D = Tuple[float, float, float]


@dataclass
class PlannerResult:
    """
    Унифицированный результат работы алгоритма
    планирования маршрута.
    """

    success: bool

    path: List[Point3D]

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    message: str = ""


class PathPlanner(NavigationAlgorithm):
    """
    Базовый интерфейс классического алгоритма
    планирования маршрута.

    Например:
    A*
    RRT
    RRT*
    PRM
    """

    algorithm_type = AlgorithmType.PATH_PLANNER

    @abstractmethod
    def plan(
        self,
        start: Point3D,
        goal: Point3D,
        environment: "EnvironmentSnapshot",
    ) -> PlannerResult:
        """
        Построить маршрут от start до goal,
        используя данные среды.
        """

        raise NotImplementedError

    def reset(self) -> None:
        """
        Большинство обычных planner'ов не имеют
        состояния между запусками.
        """

        pass