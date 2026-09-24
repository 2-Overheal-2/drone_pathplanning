from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple


Point3D = Tuple[float, float, float]
Range2D = Tuple[float, float]


@dataclass(frozen=True)
class ArenaConfig:
    """
    Конфигурация арены и генерации окружающей среды.

    Все размеры задаются в метрах.
    Максимальный размер каждой оси арены: 50 м.
    """

    # =========================================================
    # ARENA
    # =========================================================

    size: Point3D = (
        20.0,
        20.0,
        10.0,
    )

    scenario: str = "city"

    density: float = 0.15

    seed: int = 42

    start: Point3D = (
        1.5,
        1.5,
        2.0,
    )

    goal: Point3D = (
        18.0,
        18.0,
        3.0,
    )

    min_clearance: float = 0.25

    start_goal_clearance: float = 1.0

    max_objects: int = 300

    # =========================================================
    # CITY
    # =========================================================

    city_block_size: float = 6.0

    road_width: float = 1.5

    building_spacing: float = 0.30

    building_width_range: Range2D = (
        1.0,
        4.0,
    )

    building_depth_range: Range2D = (
        1.0,
        4.0,
    )

    building_height_range: Range2D = (
        2.0,
        8.0,
    )

    column_probability: float = 0.12

    column_diameter_range: Range2D = (
        0.30,
        0.80,
    )

    column_height_range: Range2D = (
        1.5,
        5.0,
    )

    # =========================================================
    # POWER LINES
    # =========================================================

    enable_power_lines: bool = False

    num_power_lines: int = 2

    power_pole_height_range: Range2D = (
        4.0,
        6.0,
    )

    power_pole_diameter: float = 0.14

    power_wire_radius: float = 0.025

    # =========================================================
    # FOREST
    # =========================================================

    tree_diameter_range: Range2D = (
        0.20,
        0.65,
    )

    tree_crown_diameter_range: Range2D = (
        1.0,
        2.5,
    )

    tree_height_range: Range2D = (
        2.0,
        7.0,
    )

    forest_cluster_probability: float = 0.65

    forest_cluster_radius: float = 3.0

    forest_cluster_count: int = 4

    forest_clearing_count: int = 1

    forest_clearing_radius: float = 2.0

    # =========================================================
    # DYNAMIC OBJECTS
    # =========================================================

    enable_dynamic_objects: bool = True

    num_cars: int = 0

    num_pedestrians: int = 0

    # =========================================================
    # VALIDATION
    # =========================================================

    def validate(
        self,
    ) -> None:
        self._validate_arena()
        self._validate_city()
        self._validate_power_lines()
        self._validate_forest()
        self._validate_dynamic()

    def _validate_arena(
        self,
    ) -> None:
        if len(self.size) != 3:
            raise ValueError(
                "size должен содержать три значения: x, y, z."
            )

        for value in self.size:
            value = float(value)

            if not (
                0.0 < value <= 50.0
            ):
                raise ValueError(
                    "Каждый размер арены должен находиться "
                    "в диапазоне (0, 50]."
                )

        if self.scenario not in {
            "city",
            "forest",
            "empty",
        }:
            raise ValueError(
                "scenario должен быть одним из: "
                "city, forest, empty."
            )

        if not (
            0.0
            <= self.density
            <= 1.0
        ):
            raise ValueError(
                "density должна быть в диапазоне [0, 1]."
            )

        if self.min_clearance < 0.0:
            raise ValueError(
                "min_clearance не может быть отрицательным."
            )

        if self.start_goal_clearance < 0.0:
            raise ValueError(
                "start_goal_clearance не может быть отрицательным."
            )

        if self.max_objects <= 0:
            raise ValueError(
                "max_objects должен быть > 0."
            )

        self._validate_point(
            "start",
            self.start,
        )

        self._validate_point(
            "goal",
            self.goal,
        )

    def _validate_city(
        self,
    ) -> None:
        if self.city_block_size <= 0.0:
            raise ValueError(
                "city_block_size должен быть > 0."
            )

        if self.road_width < 0.0:
            raise ValueError(
                "road_width не может быть отрицательным."
            )

        if self.building_spacing < 0.0:
            raise ValueError(
                "building_spacing не может быть отрицательным."
            )

        if not (
            0.0
            <= self.column_probability
            <= 1.0
        ):
            raise ValueError(
                "column_probability должна быть в диапазоне [0, 1]."
            )

        self._validate_range(
            "building_width_range",
            self.building_width_range,
        )

        self._validate_range(
            "building_depth_range",
            self.building_depth_range,
        )

        self._validate_range(
            "building_height_range",
            self.building_height_range,
        )

        self._validate_range(
            "column_diameter_range",
            self.column_diameter_range,
        )

        self._validate_range(
            "column_height_range",
            self.column_height_range,
        )

    def _validate_power_lines(
        self,
    ) -> None:
        if self.num_power_lines < 0:
            raise ValueError(
                "num_power_lines не может быть отрицательным."
            )

        if self.power_pole_diameter <= 0.0:
            raise ValueError(
                "power_pole_diameter должен быть > 0."
            )

        if self.power_wire_radius <= 0.0:
            raise ValueError(
                "power_wire_radius должен быть > 0."
            )

        self._validate_range(
            "power_pole_height_range",
            self.power_pole_height_range,
        )

    def _validate_forest(
        self,
    ) -> None:
        self._validate_range(
            "tree_diameter_range",
            self.tree_diameter_range,
        )

        self._validate_range(
            "tree_crown_diameter_range",
            self.tree_crown_diameter_range,
        )

        self._validate_range(
            "tree_height_range",
            self.tree_height_range,
        )

        if not (
            0.0
            <= self.forest_cluster_probability
            <= 1.0
        ):
            raise ValueError(
                "forest_cluster_probability должна быть "
                "в диапазоне [0, 1]."
            )

        if self.forest_cluster_radius <= 0.0:
            raise ValueError(
                "forest_cluster_radius должен быть > 0."
            )

        if self.forest_cluster_count < 0:
            raise ValueError(
                "forest_cluster_count не может быть отрицательным."
            )

        if self.forest_clearing_count < 0:
            raise ValueError(
                "forest_clearing_count не может быть отрицательным."
            )

        if self.forest_clearing_radius < 0.0:
            raise ValueError(
                "forest_clearing_radius не может быть отрицательным."
            )

    def _validate_dynamic(
        self,
    ) -> None:
        if self.num_cars < 0:
            raise ValueError(
                "num_cars не может быть отрицательным."
            )

        if self.num_pedestrians < 0:
            raise ValueError(
                "num_pedestrians не может быть отрицательным."
            )

    def _validate_point(
        self,
        name: str,
        point: Point3D,
    ) -> None:
        if len(point) != 3:
            raise ValueError(
                f"{name} должен содержать 3 координаты."
            )

        x = float(point[0])
        y = float(point[1])
        z = float(point[2])

        size_x = float(
            self.size[0]
        )

        size_y = float(
            self.size[1]
        )

        size_z = float(
            self.size[2]
        )

        if not (
            0.0 <= x <= size_x
            and 0.0 <= y <= size_y
            and 0.0 <= z <= size_z
        ):
            raise ValueError(
                f"{name}={point} находится за пределами "
                f"арены size={self.size}."
            )

    @staticmethod
    def _validate_range(
        name: str,
        value_range: Range2D,
    ) -> None:
        if len(value_range) != 2:
            raise ValueError(
                f"{name} должен содержать два значения."
            )

        minimum = float(
            value_range[0]
        )

        maximum = float(
            value_range[1]
        )

        if minimum <= 0.0:
            raise ValueError(
                f"{name}: minimum должен быть > 0."
            )

        if maximum < minimum:
            raise ValueError(
                f"{name}: maximum должен быть >= minimum."
            )

    def with_seed(
        self,
        seed: int,
    ) -> "ArenaConfig":
        return replace(
            self,
            seed=int(seed),
        )