from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple
import math

import numpy as np
import pybullet as p


Vector3 = Sequence[float]
Quaternion = Sequence[float]


def _rgba(value: Sequence[float]) -> List[float]:
    rgba = list(value)
    if len(rgba) != 4:
        raise ValueError("Цвет должен быть RGBA из 4 компонентов.")
    return [float(x) for x in rgba]


@dataclass
class Obstacle:
    """Нейтральное описание статического препятствия + его PyBullet body."""

    kind: str
    geometry: str
    position: np.ndarray
    dimensions: np.ndarray
    color: Sequence[float] = (0.6, 0.6, 0.6, 1.0)
    body_id: Optional[int] = None

    @property
    def footprint_area(self) -> float:
        if self.geometry == "cylinder":
            radius = float(self.dimensions[0]) / 2.0
            return math.pi * radius * radius
        return float(self.dimensions[0] * self.dimensions[1])

    @property
    def aabb(self) -> Tuple[List[float], List[float]]:
        # Для planner'а и быстрых проверок цилиндр тоже представляем AABB.
        half = self.dimensions / 2.0
        return (self.position - half).tolist(), (self.position + half).tolist()

    def spawn(self, client_id: int) -> int:
        collision, visual = self._create_shapes(client_id)

        self.body_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=self.position.tolist(),
            physicsClientId=client_id,
        )
        return self.body_id

    def clear_body_id(self) -> None:
        self.body_id = None

    def _create_shapes(self, client_id: int) -> Tuple[int, int]:
        if self.geometry == "box":
            half = (self.dimensions / 2.0).tolist()
            collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=half,
                physicsClientId=client_id,
            )
            visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=half,
                rgbaColor=_rgba(self.color),
                physicsClientId=client_id,
            )
            return collision, visual

        if self.geometry == "cylinder":
            diameter, _, height = self.dimensions.tolist()
            radius = diameter / 2.0
            collision = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=radius,
                height=height,
                physicsClientId=client_id,
            )
            visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=radius,
                length=height,
                rgbaColor=_rgba(self.color),
                physicsClientId=client_id,
            )
            return collision, visual

        raise ValueError(f"Неподдерживаемая геометрия препятствия: {self.geometry}")

    def export(self) -> dict:
        aabb_min, aabb_max = self.aabb
        return {
            "type": self.geometry,
            "kind": self.kind,
            "position": self.position.tolist(),
            "dimensions": self.dimensions.tolist(),
            "aabb_min": aabb_min,
            "aabb_max": aabb_max,
            "dynamic": False,
            "body_id": self.body_id,
        }


class Trajectory:
    """Интерфейс кинематической траектории."""

    def sample(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class LinearTrajectory(Trajectory):
    """Движение start -> end -> start с заданной скоростью."""

    def __init__(self, start: Vector3, end: Vector3, speed: float = 1.0, loop: bool = True):
        self.start = np.asarray(start, dtype=float)
        self.end = np.asarray(end, dtype=float)
        self.speed = max(0.0, float(speed))
        self.loop = bool(loop)

        delta = self.end - self.start
        self.length = float(np.linalg.norm(delta))
        self.direction = delta / self.length if self.length > 1e-9 else np.zeros(3, dtype=float)

    def sample(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        if self.length <= 1e-9:
            position = self.start.copy()
        else:
            distance = self.speed * max(0.0, float(t))

            if self.loop:
                phase = (distance / self.length) % 2.0
                if phase <= 1.0:
                    position = self.start + self.direction * (phase * self.length)
                    direction = self.direction
                else:
                    position = self.end - self.direction * ((phase - 1.0) * self.length)
                    direction = -self.direction
            else:
                distance = min(distance, self.length)
                position = self.start + self.direction * distance
                direction = self.direction

            yaw = math.atan2(direction[1], direction[0]) if np.linalg.norm(direction[:2]) > 1e-9 else 0.0
            return position, np.asarray(p.getQuaternionFromEuler([0.0, 0.0, yaw]), dtype=float)

        return position, np.asarray([0.0, 0.0, 0.0, 1.0], dtype=float)


class WaypointTrajectory(Trajectory):
    """Равномерное движение по последовательности 3D waypoint'ов."""

    def __init__(self, waypoints: Sequence[Vector3], speed: float = 1.0, loop: bool = True):
        self.waypoints = [np.asarray(point, dtype=float) for point in waypoints]
        self.speed = max(0.0, float(speed))
        self.loop = bool(loop)

        if len(self.waypoints) < 2:
            raise ValueError("WaypointTrajectory требует минимум 2 точки.")

        points = self.waypoints + ([self.waypoints[0]] if self.loop else [])
        self.segment_lengths = [
            float(np.linalg.norm(b - a))
            for a, b in zip(points[:-1], points[1:])
        ]
        self.total_length = float(sum(self.segment_lengths))

    def sample(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        if self.total_length <= 1e-9:
            return self.waypoints[0].copy(), np.asarray([0.0, 0.0, 0.0, 1.0], dtype=float)

        distance = self.speed * max(0.0, float(t))
        distance = distance % self.total_length if self.loop else min(distance, self.total_length)

        points = self.waypoints + ([self.waypoints[0]] if self.loop else [])

        for index, segment_length in enumerate(self.segment_lengths):
            if distance <= segment_length or index == len(self.segment_lengths) - 1:
                a, b = points[index], points[index + 1]
                alpha = 0.0 if segment_length <= 1e-9 else distance / segment_length
                position = a + (b - a) * alpha
                direction = b - a
                yaw = math.atan2(direction[1], direction[0]) if np.linalg.norm(direction[:2]) > 1e-9 else 0.0
                orientation = np.asarray(p.getQuaternionFromEuler([0.0, 0.0, yaw]), dtype=float)
                return position, orientation

            distance -= segment_length

        return self.waypoints[-1].copy(), np.asarray([0.0, 0.0, 0.0, 1.0], dtype=float)


@dataclass
class DynamicObject:
    """Кинематический объект: машина, пешеход и т.п."""

    kind: str
    geometry: str
    dimensions: np.ndarray
    trajectory: Trajectory
    color: Sequence[float] = (0.2, 0.2, 0.8, 1.0)

    body_id: Optional[int] = None
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    orientation: np.ndarray = field(
        default_factory=lambda: np.asarray([0.0, 0.0, 0.0, 1.0], dtype=float)
    )

    @property
    def aabb(self) -> Tuple[List[float], List[float]]:
        half = self.dimensions / 2.0
        return (self.position - half).tolist(), (self.position + half).tolist()

    def spawn(self, client_id: int, t0: float = 0.0) -> int:
        self.position, self.orientation = self.trajectory.sample(t0)

        if self.geometry == "box":
            half = (self.dimensions / 2.0).tolist()
            collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=half,
                physicsClientId=client_id,
            )
            visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=half,
                rgbaColor=_rgba(self.color),
                physicsClientId=client_id,
            )
        elif self.geometry == "cylinder":
            diameter, _, height = self.dimensions.tolist()
            radius = diameter / 2.0
            collision = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=radius,
                height=height,
                physicsClientId=client_id,
            )
            visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=radius,
                length=height,
                rgbaColor=_rgba(self.color),
                physicsClientId=client_id,
            )
        else:
            raise ValueError(f"Неподдерживаемая геометрия dynamic object: {self.geometry}")

        self.body_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=self.position.tolist(),
            baseOrientation=self.orientation.tolist(),
            physicsClientId=client_id,
        )
        return self.body_id

    def update(self, t: float, client_id: int) -> None:
        if self.body_id is None:
            return

        self.position, self.orientation = self.trajectory.sample(t)
        p.resetBasePositionAndOrientation(
            self.body_id,
            self.position.tolist(),
            self.orientation.tolist(),
            physicsClientId=client_id,
        )

    def clear_body_id(self) -> None:
        self.body_id = None

    def export(self) -> dict:
        aabb_min, aabb_max = self.aabb
        return {
            "type": self.geometry,
            "kind": self.kind,
            "position": self.position.tolist(),
            "dimensions": self.dimensions.tolist(),
            "aabb_min": aabb_min,
            "aabb_max": aabb_max,
            "dynamic": True,
            "body_id": self.body_id,
        }
