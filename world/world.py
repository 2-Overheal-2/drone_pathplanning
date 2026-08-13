from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pybullet as p

from world.config import ArenaConfig
from world.objects import (
    DynamicObject,
    LinearTrajectory,
    Obstacle,
    WaypointTrajectory,
)
from world.scenarios import make_scenario


@dataclass(frozen=True)
class EnvironmentSnapshot:
    """Нейтральные данные среды для planner'ов и других модулей."""

    bounds: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    start: Tuple[float, float, float]
    goal: Tuple[float, float, float]
    static_obstacles: list
    dynamic_obstacles: list


class SimulationWorld:
    """
    Мир не наследуется от Gymnasium.

    Он отвечает только за:
    - генерацию объектов;
    - создание их в PyBullet;
    - обновление динамических объектов;
    - экспорт EnvironmentSnapshot.
    """

    def __init__(self, config: ArenaConfig, client_id: int):
        config.validate()

        self.config = config
        self.client_id = int(client_id)
        self.rng = np.random.default_rng(config.seed)

        self.static_obstacles: List[Obstacle] = []
        self.dynamic_objects: List[DynamicObject] = []
        self.floor_id: Optional[int] = None
        self._generated = False

    def generate(self) -> "SimulationWorld":
        scenario = make_scenario(self.config.scenario)
        self.static_obstacles = scenario.generate(self.config, self.rng)
        self.dynamic_objects = self._generate_dynamic_objects()
        self._generated = True
        return self

    def spawn(self, create_floor: bool = True) -> None:
        if not self._generated:
            self.generate()

        self.clear_body_ids()

        if create_floor:
            self._spawn_floor()

        for obstacle in self.static_obstacles:
            obstacle.spawn(self.client_id)

        for obj in self.dynamic_objects:
            obj.spawn(self.client_id, t0=0.0)

    def update(self, simulation_time: float) -> None:
        for obj in self.dynamic_objects:
            obj.update(simulation_time, self.client_id)

    def clear_body_ids(self) -> None:
        self.floor_id = None
        for obstacle in self.static_obstacles:
            obstacle.clear_body_id()
        for obj in self.dynamic_objects:
            obj.clear_body_id()

    def all_body_ids(self) -> List[int]:
        ids: List[int] = []
        if self.floor_id is not None:
            ids.append(self.floor_id)
        ids.extend(
            obstacle.body_id
            for obstacle in self.static_obstacles
            if obstacle.body_id is not None
        )
        ids.extend(
            obj.body_id
            for obj in self.dynamic_objects
            if obj.body_id is not None
        )
        return ids

    def snapshot(self) -> EnvironmentSnapshot:
        sx, sy, sz = self.config.size

        return EnvironmentSnapshot(
            bounds=((0.0, 0.0, 0.0), (sx, sy, sz)),
            start=tuple(self.config.start),
            goal=tuple(self.config.goal),
            static_obstacles=[obstacle.export() for obstacle in self.static_obstacles],
            dynamic_obstacles=[obj.export() for obj in self.dynamic_objects],
        )

    def _spawn_floor(self) -> None:
        sx, sy, _ = self.config.size
        half_height = 0.05

        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[sx / 2.0, sy / 2.0, half_height],
            physicsClientId=self.client_id,
        )
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[sx / 2.0, sy / 2.0, half_height],
            rgbaColor=[0.72, 0.72, 0.72, 1.0],
            physicsClientId=self.client_id,
        )

        self.floor_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[sx / 2.0, sy / 2.0, -half_height],
            physicsClientId=self.client_id,
        )

    def _generate_dynamic_objects(self) -> List[DynamicObject]:
        if not self.config.enable_dynamic_objects:
            return []

        sx, sy, _ = self.config.size
        result: List[DynamicObject] = []

        # Машины идут по простым горизонтальным линиям.
        for i in range(self.config.num_cars):
            y = sy * (0.20 + 0.12 * (i % 5))
            y = min(max(y, 0.6), max(0.6, sy - 0.6))

            result.append(
                DynamicObject(
                    kind="car",
                    geometry="box",
                    dimensions=np.asarray([1.8, 0.9, 0.8], dtype=float),
                    trajectory=LinearTrajectory(
                        start=[0.8, y, 0.4],
                        end=[max(0.8, sx - 0.8), y, 0.4],
                        speed=1.5 + 0.3 * i,
                        loop=True,
                    ),
                    color=(0.10, 0.25, 0.80, 1.0),
                )
            )

        # Пешеходы идут по прямоугольным waypoint-траекториям.
        for i in range(self.config.num_pedestrians):
            x1 = min(max(sx * (0.25 + 0.08 * i), 0.5), max(0.5, sx - 0.5))
            x2 = min(max(x1 + max(0.5, sx * 0.10), 0.5), max(0.5, sx - 0.5))
            y1 = min(0.8, sy / 2.0)
            y2 = max(y1, sy - 0.8)

            result.append(
                DynamicObject(
                    kind="pedestrian",
                    geometry="cylinder",
                    dimensions=np.asarray([0.45, 0.45, 1.7], dtype=float),
                    trajectory=WaypointTrajectory(
                        waypoints=[
                            [x1, y1, 0.85],
                            [x1, y2, 0.85],
                            [x2, y2, 0.85],
                            [x2, y1, 0.85],
                        ],
                        speed=1.0 + 0.1 * i,
                        loop=True,
                    ),
                    color=(0.85, 0.45, 0.15, 1.0),
                )
            )

        return result
