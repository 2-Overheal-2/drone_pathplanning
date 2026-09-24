from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pybullet as p

from world.models.base import ObjectModel, PrimitiveModel


def _as_vector3(
    value: Sequence[float],
) -> np.ndarray:
    array = np.asarray(
        value,
        dtype=float,
    )

    if array.shape != (3,):
        raise ValueError(
            "Ожидался вектор из трёх значений."
        )

    return array


def _as_quaternion(
    value: Sequence[float],
) -> np.ndarray:
    array = np.asarray(
        value,
        dtype=float,
    )

    if array.shape != (4,):
        raise ValueError(
            "Quaternion должен содержать 4 значения."
        )

    return array


def _oriented_aabb(
    position: np.ndarray,
    dimensions: np.ndarray,
    orientation: np.ndarray,
):
    """
    Консервативный AABB объекта
    с учётом ориентации.
    """

    half = (
        dimensions
        / 2.0
    )

    rotation = np.asarray(
        p.getMatrixFromQuaternion(
            orientation.tolist()
        ),
        dtype=float,
    ).reshape(
        3,
        3,
    )

    world_half = (
        np.abs(rotation)
        @ half
    )

    return (
        position - world_half,
        position + world_half,
    )


@dataclass
class Obstacle:
    """
    Статический объект среды.

    Старый режим:

        geometry="box"
        dimensions=[...]

    Новый режим:

        model=BuildingModel(...)
        model=TreeModel(...)
        model=WireModel(...)

    Если model задан, именно модель создаёт
    визуальную и collision геометрию.
    """

    kind: str

    geometry: str

    position: np.ndarray

    dimensions: np.ndarray

    color: Sequence[float] = (
        0.6,
        0.6,
        0.6,
        1.0,
    )

    orientation: np.ndarray = field(
        default_factory=lambda: np.asarray(
            [
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            dtype=float,
        )
    )

    model: ObjectModel | None = None

    body_id: int | None = None

    def __post_init__(
        self,
    ) -> None:
        self.position = (
            _as_vector3(
                self.position
            )
        )

        self.orientation = (
            _as_quaternion(
                self.orientation
            )
        )

        self.dimensions = (
            _as_vector3(
                self.dimensions
            )
        )

        self.color = tuple(
            float(value)
            for value
            in self.color
        )

        if self.model is not None:
            self.dimensions = (
                self.model
                .bounding_dimensions
                .astype(float)
            )

        if np.any(
            self.dimensions <= 0.0
        ):
            raise ValueError(
                "Размеры препятствия должны быть > 0."
            )

    @property
    def footprint_area(
        self,
    ) -> float:
        """
        Приблизительная площадь объекта на XY.

        Используется генераторами сценариев
        для контроля density.
        """

        width = float(
            self.dimensions[0]
        )

        depth = float(
            self.dimensions[1]
        )

        if self.geometry == "cylinder":
            radius_x = (
                width
                / 2.0
            )

            radius_y = (
                depth
                / 2.0
            )

            return float(
                np.pi
                * radius_x
                * radius_y
            )

        return float(
            width
            * depth
        )

    @property
    def aabb(
        self,
    ):
        """
        Получить AABB без передачи body_id наружу.
        """

        if self.model is not None:
            return self.model.world_aabb(
                position=self.position,
                orientation=self.orientation,
            )

        return _oriented_aabb(
            position=self.position,
            dimensions=self.dimensions,
            orientation=self.orientation,
        )

    def spawn(
        self,
        client_id: int,
    ) -> int:
        """
        Создать статический объект в PyBullet.
        """

        if self.body_id is not None:
            return self.body_id

        if self.model is not None:
            self.body_id = (
                self.model.spawn(
                    client_id=client_id,
                    position=self.position,
                    orientation=self.orientation,
                    mass=0.0,
                )
            )

            return self.body_id

        primitive = PrimitiveModel(
            kind=self.kind,
            geometry=self.geometry,
            dimensions=self.dimensions,
            color=self.color,
        )

        self.body_id = (
            primitive.spawn(
                client_id=client_id,
                position=self.position,
                orientation=self.orientation,
                mass=0.0,
            )
        )

        return self.body_id

    def export(
        self,
    ) -> dict:
        """
        Нейтральное описание объекта.

        body_id сюда специально не попадает.
        """

        aabb_min, aabb_max = (
            self.aabb
        )

        return {
            "type": self.geometry,
            "kind": self.kind,

            "model": (
                self.model.__class__.__name__
                if self.model is not None
                else None
            ),

            "position": (
                self.position.tolist()
            ),

            "dimensions": (
                self.dimensions.tolist()
            ),

            "orientation": (
                self.orientation.tolist()
            ),

            "aabb_min": (
                aabb_min.tolist()
            ),

            "aabb_max": (
                aabb_max.tolist()
            ),

            "dynamic": False,
        }


class Trajectory(ABC):
    """
    Базовый интерфейс траектории.
    """

    @abstractmethod
    def sample(
        self,
        simulation_time: float,
    ):
        """
        Вернуть:

            position,
            orientation
        """

        raise NotImplementedError


class LinearTrajectory(
    Trajectory
):
    """
    Линейное движение между двумя точками.

    При loop=True:

        start -> end -> start -> end ...
    """

    def __init__(
        self,
        start: Sequence[float],
        end: Sequence[float],
        speed: float,
        loop: bool = True,
    ):
        self.start = (
            _as_vector3(
                start
            )
        )

        self.end = (
            _as_vector3(
                end
            )
        )

        self.speed = float(
            speed
        )

        self.loop = bool(
            loop
        )

        if self.speed < 0.0:
            raise ValueError(
                "speed не может быть отрицательной."
            )

        self.direction = (
            self.end
            - self.start
        )

        self.distance = float(
            np.linalg.norm(
                self.direction
            )
        )

        if self.distance > 1e-9:
            self.unit_direction = (
                self.direction
                / self.distance
            )

        else:
            self.unit_direction = (
                np.zeros(
                    3,
                    dtype=float,
                )
            )

    def sample(
        self,
        simulation_time: float,
    ):
        time_value = max(
            0.0,
            float(
                simulation_time
            ),
        )

        if (
            self.distance <= 1e-9
            or self.speed <= 1e-9
        ):
            return (
                self.start.copy(),

                np.asarray(
                    [
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    dtype=float,
                ),
            )

        travelled = (
            time_value
            * self.speed
        )

        if self.loop:
            cycle_length = (
                2.0
                * self.distance
            )

            cycle_position = (
                travelled
                % cycle_length
            )

            if (
                cycle_position
                <= self.distance
            ):
                distance_along = (
                    cycle_position
                )

                movement_direction = (
                    self.unit_direction
                )

            else:
                distance_along = (
                    cycle_length
                    - cycle_position
                )

                movement_direction = (
                    -self.unit_direction
                )

        else:
            distance_along = min(
                travelled,
                self.distance,
            )

            movement_direction = (
                self.unit_direction
            )

        position = (
            self.start
            + self.unit_direction
            * distance_along
        )

        yaw = float(
            np.arctan2(
                movement_direction[1],
                movement_direction[0],
            )
        )

        orientation = np.asarray(
            p.getQuaternionFromEuler(
                [
                    0.0,
                    0.0,
                    yaw,
                ]
            ),
            dtype=float,
        )

        return (
            position,
            orientation,
        )


class WaypointTrajectory(
    Trajectory
):
    """
    Движение через последовательность
    трёхмерных точек.
    """

    def __init__(
        self,
        waypoints: Sequence[
            Sequence[float]
        ],
        speed: float,
        loop: bool = True,
    ):
        if len(
            waypoints
        ) < 2:
            raise ValueError(
                "WaypointTrajectory требует минимум две точки."
            )

        self.waypoints = [
            _as_vector3(
                point
            )
            for point
            in waypoints
        ]

        self.speed = float(
            speed
        )

        self.loop = bool(
            loop
        )

        if self.speed < 0.0:
            raise ValueError(
                "speed не может быть отрицательной."
            )

        self._segments = []

        for index in range(
            len(self.waypoints) - 1
        ):
            start = (
                self.waypoints[index]
            )

            end = (
                self.waypoints[
                    index + 1
                ]
            )

            vector = (
                end - start
            )

            length = float(
                np.linalg.norm(
                    vector
                )
            )

            if length <= 1e-9:
                continue

            self._segments.append(
                (
                    start,
                    end,
                    vector / length,
                    length,
                )
            )

        if not self._segments:
            raise ValueError(
                "Все waypoints совпадают."
            )

        self.total_length = float(
            sum(
                segment[3]
                for segment
                in self._segments
            )
        )

    def sample(
        self,
        simulation_time: float,
    ):
        time_value = max(
            0.0,
            float(
                simulation_time
            ),
        )

        if self.speed <= 1e-9:
            position = (
                self._segments[0][0]
                .copy()
            )

            orientation = (
                self._orientation_from_direction(
                    self._segments[0][2]
                )
            )

            return (
                position,
                orientation,
            )

        travelled = (
            time_value
            * self.speed
        )

        if self.loop:
            travelled = (
                travelled
                % self.total_length
            )

        else:
            travelled = min(
                travelled,
                self.total_length,
            )

        remaining = (
            travelled
        )

        for (
            start,
            end,
            direction,
            length,
        ) in self._segments:
            if remaining <= length:
                position = (
                    start
                    + direction
                    * remaining
                )

                orientation = (
                    self._orientation_from_direction(
                        direction
                    )
                )

                return (
                    position,
                    orientation,
                )

            remaining -= (
                length
            )

        (
            start,
            end,
            direction,
            length,
        ) = self._segments[-1]

        return (
            end.copy(),

            self._orientation_from_direction(
                direction
            ),
        )

    @staticmethod
    def _orientation_from_direction(
        direction: np.ndarray,
    ) -> np.ndarray:
        yaw = float(
            np.arctan2(
                direction[1],
                direction[0],
            )
        )

        return np.asarray(
            p.getQuaternionFromEuler(
                [
                    0.0,
                    0.0,
                    yaw,
                ]
            ),
            dtype=float,
        )


@dataclass
class DynamicObject:
    """
    Кинематический динамический объект.

    Примеры:

        car
        pedestrian

    Объект перемещается по Trajectory через
    resetBasePositionAndOrientation().

    mass=0 используется намеренно.
    """

    kind: str

    geometry: str

    dimensions: np.ndarray

    trajectory: Trajectory

    color: Sequence[float] = (
        0.8,
        0.3,
        0.2,
        1.0,
    )

    #
    # Вот этого поля не было
    # в твоей текущей версии.
    #
    model: ObjectModel | None = None

    body_id: int | None = None

    position: np.ndarray = field(
        default_factory=lambda: np.zeros(
            3,
            dtype=float,
        )
    )

    orientation: np.ndarray = field(
        default_factory=lambda: np.asarray(
            [
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            dtype=float,
        )
    )

    def __post_init__(
        self,
    ) -> None:
        self.dimensions = (
            _as_vector3(
                self.dimensions
            )
        )

        self.color = tuple(
            float(value)
            for value
            in self.color
        )

        if self.model is not None:
            self.dimensions = (
                self.model
                .bounding_dimensions
                .astype(float)
            )

        (
            initial_position,
            initial_orientation,
        ) = self.trajectory.sample(
            0.0
        )

        self.position = (
            _as_vector3(
                initial_position
            )
        )

        self.orientation = (
            _as_quaternion(
                initial_orientation
            )
        )

        if np.any(
            self.dimensions <= 0.0
        ):
            raise ValueError(
                "Размеры DynamicObject должны быть > 0."
            )

    @property
    def aabb(
        self,
    ):
        """
        Текущий AABB динамического объекта.
        """

        if self.model is not None:
            return self.model.world_aabb(
                position=self.position,
                orientation=self.orientation,
            )

        return _oriented_aabb(
            position=self.position,
            dimensions=self.dimensions,
            orientation=self.orientation,
        )

    def spawn(
        self,
        client_id: int,
    ) -> int:
        """
        Создать динамический объект.

        Для CarModel:
            visual = OBJ
            collision = box

        Для старых объектов:
            используется PrimitiveModel.
        """

        (
            position,
            orientation,
        ) = self.trajectory.sample(
            0.0
        )

        self.position = (
            _as_vector3(
                position
            )
        )

        self.orientation = (
            _as_quaternion(
                orientation
            )
        )

        if self.model is not None:
            self.body_id = (
                self.model.spawn(
                    client_id=client_id,
                    position=self.position,
                    orientation=self.orientation,
                    mass=0.0,
                )
            )

            return self.body_id

        primitive = PrimitiveModel(
            kind=self.kind,
            geometry=self.geometry,
            dimensions=self.dimensions,
            color=self.color,
        )

        self.body_id = (
            primitive.spawn(
                client_id=client_id,
                position=self.position,
                orientation=self.orientation,
                mass=0.0,
            )
        )

        return self.body_id

    def update(
        self,
        simulation_time: float,
        client_id: int,
    ) -> None:
        """
        Переместить объект в положение,
        заданное траекторией.
        """

        (
            position,
            orientation,
        ) = self.trajectory.sample(
            simulation_time
        )

        self.position = (
            _as_vector3(
                position
            )
        )

        self.orientation = (
            _as_quaternion(
                orientation
            )
        )

        if self.body_id is None:
            return

        p.resetBasePositionAndOrientation(
            self.body_id,
            self.position.tolist(),
            self.orientation.tolist(),
            physicsClientId=client_id,
        )

    def export(
        self,
    ) -> dict:
        """
        Нейтральное описание объекта.

        body_id наружу не передаётся.
        """

        aabb_min, aabb_max = (
            self.aabb
        )

        return {
            "type": self.geometry,
            "kind": self.kind,

            "model": (
                self.model.__class__.__name__
                if self.model is not None
                else None
            ),

            "position": (
                self.position.tolist()
            ),

            "dimensions": (
                self.dimensions.tolist()
            ),

            "orientation": (
                self.orientation.tolist()
            ),

            "aabb_min": (
                aabb_min.tolist()
            ),

            "aabb_max": (
                aabb_max.tolist()
            ),

            "dynamic": True,
        }