from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p

from world.config import ArenaConfig

from world.models import CarModel

from world.objects import (
    DynamicObject,
    LinearTrajectory,
    Obstacle,
    WaypointTrajectory,
)

from world.scenarios import make_scenario


Point3D = Tuple[
    float,
    float,
    float,
]


@dataclass(frozen=True)
class EnvironmentSnapshot:
    """
    Нейтральное описание текущего состояния среды.

    Этот объект используется как контракт между:

        SimulationWorld
            ↓
        AlgorithmAdapter
            ↓
        PathPlanner / PolicyController

    PyBullet body_id сюда намеренно не передаются.
    """

    bounds: Tuple[
        Point3D,
        Point3D,
    ]

    start: Point3D
    goal: Point3D

    static_obstacles: list
    dynamic_obstacles: list


class SimulationWorld:
    """
    Основной класс окружающего мира.

    Отвечает за:

    - генерацию сценария;
    - создание объектов в PyBullet;
    - динамические объекты;
    - EnvironmentSnapshot;
    - collision queries;
    - reset среды.

    Не отвечает за:

    - управление дроном;
    - PPO;
    - A* / RRT / PRM;
    - reward;
    - action space.
    """

    def __init__(
        self,
        config: ArenaConfig,
        client_id: int,
    ):
        config.validate()

        self.config = config

        self.client_id = int(
            client_id
        )

        self.static_obstacles: List[
            Obstacle
        ] = []

        self.dynamic_objects: List[
            DynamicObject
        ] = []

        self.floor_id: Optional[
            int
        ] = None

        self.generated = False
        self.spawned = False

    # =========================================================
    # GENERATION
    # =========================================================

    def generate(
        self,
    ) -> "SimulationWorld":
        """
        Сгенерировать логическое описание мира.

        На этом этапе PyBullet bodies
        ещё не создаются.
        """

        self.config.validate()

        rng = np.random.default_rng(
            self.config.seed
        )

        scenario = make_scenario(
            self.config.scenario
        )

        self.static_obstacles = (
            scenario.generate(
                self.config,
                rng,
            )
        )

        self.dynamic_objects = (
            self._generate_dynamic_objects()
        )

        self.generated = True
        self.spawned = False

        return self

    # =========================================================
    # SPAWN
    # =========================================================

    def spawn(
        self,
        create_floor: bool = True,
    ) -> None:
        """
        Создать все объекты мира в PyBullet.

        create_floor=True:
            standalone SimulationWorld.

        create_floor=False:
            используется внутри Base3DEnv,
            потому что BaseAviary уже создаёт plane.
        """

        if not self.generated:
            self.generate()

        if create_floor:
            self._spawn_floor()

        for obstacle in (
            self.static_obstacles
        ):
            obstacle.spawn(
                self.client_id
            )

        for dynamic_object in (
            self.dynamic_objects
        ):
            dynamic_object.spawn(
                self.client_id
            )

        self.spawned = True

    # =========================================================
    # UPDATE
    # =========================================================

    def update(
        self,
        simulation_time: float,
    ) -> None:
        """
        Обновить динамические объекты.

        Статические объекты не изменяются.
        """

        if not self.spawned:
            return

        for dynamic_object in (
            self.dynamic_objects
        ):
            dynamic_object.update(
                simulation_time,
                self.client_id,
            )

    # =========================================================
    # SNAPSHOT
    # =========================================================

    def snapshot(
        self,
    ) -> EnvironmentSnapshot:
        """
        Получить нейтральное описание текущего мира.

        Snapshot можно безопасно передавать
        внешнему алгоритму навигации.
        """

        size_x = float(
            self.config.size[0]
        )

        size_y = float(
            self.config.size[1]
        )

        size_z = float(
            self.config.size[2]
        )

        return EnvironmentSnapshot(
            bounds=(
                (
                    0.0,
                    0.0,
                    0.0,
                ),
                (
                    size_x,
                    size_y,
                    size_z,
                ),
            ),

            start=tuple(
                float(value)
                for value
                in self.config.start
            ),

            goal=tuple(
                float(value)
                for value
                in self.config.goal
            ),

            static_obstacles=[
                obstacle.export()
                for obstacle
                in self.static_obstacles
            ],

            dynamic_obstacles=[
                dynamic_object.export()
                for dynamic_object
                in self.dynamic_objects
            ],
        )

    # =========================================================
    # RESET
    # =========================================================

    def reset(
        self,
        regenerate: bool = False,
        create_floor: bool = True,
        pybullet_was_reset: bool = False,
    ) -> None:
        """
        Перезапустить окружающую среду.

        regenerate=False:
            использовать старое расположение объектов.

        regenerate=True:
            сгенерировать новую сцену.

        pybullet_was_reset=False:
            SimulationWorld сам удаляет свои bodies.

        pybullet_was_reset=True:
            BaseAviary уже вызвал resetSimulation().

            В этом случае старые body_id уже
            не существуют, поэтому удалять их нельзя.
        """

        if pybullet_was_reset:
            self._clear_body_ids()

            self.spawned = False

        else:
            self.remove_bodies()

        if regenerate:
            self.generate()

        self.spawn(
            create_floor=create_floor
        )

    # =========================================================
    # REMOVE
    # =========================================================

    def remove_bodies(
        self,
    ) -> None:
        """
        Удалить из PyBullet только объекты,
        принадлежащие SimulationWorld.

        Дрон этим методом не удаляется.
        """

        if not p.isConnected(
            self.client_id
        ):
            self._clear_body_ids()

            self.spawned = False

            return

        existing_body_ids = {
            p.getBodyUniqueId(
                index,
                physicsClientId=(
                    self.client_id
                ),
            )
            for index
            in range(
                p.getNumBodies(
                    physicsClientId=(
                        self.client_id
                    ),
                )
            )
        }

        body_ids = []

        if self.floor_id is not None:
            body_ids.append(
                self.floor_id
            )

        for obstacle in (
            self.static_obstacles
        ):
            if obstacle.body_id is not None:
                body_ids.append(
                    obstacle.body_id
                )

        for dynamic_object in (
            self.dynamic_objects
        ):
            if (
                dynamic_object.body_id
                is not None
            ):
                body_ids.append(
                    dynamic_object.body_id
                )

        for body_id in body_ids:
            if body_id in existing_body_ids:
                p.removeBody(
                    body_id,
                    physicsClientId=(
                        self.client_id
                    ),
                )

        self._clear_body_ids()

        self.spawned = False

    def _clear_body_ids(
        self,
    ) -> None:
        """
        Забыть PyBullet body_id.

        Этот метод ничего физически
        из PyBullet не удаляет.
        """

        self.floor_id = None

        for obstacle in (
            self.static_obstacles
        ):
            obstacle.body_id = None

        for dynamic_object in (
            self.dynamic_objects
        ):
            dynamic_object.body_id = None

    # =========================================================
    # COLLISION QUERIES
    # =========================================================

    def distance_to_nearest_obstacle(
        self,
        position: Sequence[float],
    ) -> float:
        """
        Расстояние от точки до ближайшего
        статического или динамического препятствия.

        Пока используется AABB.
        """

        point = np.asarray(
            position,
            dtype=float,
        )

        if point.shape != (3,):
            raise ValueError(
                "position должен содержать 3 координаты."
            )

        minimum_distance = float(
            "inf"
        )

        snapshot = self.snapshot()

        obstacles = (
            snapshot.static_obstacles
            + snapshot.dynamic_obstacles
        )

        for obstacle in obstacles:
            aabb_min = np.asarray(
                obstacle["aabb_min"],
                dtype=float,
            )

            aabb_max = np.asarray(
                obstacle["aabb_max"],
                dtype=float,
            )

            closest_point = np.minimum(
                np.maximum(
                    point,
                    aabb_min,
                ),
                aabb_max,
            )

            distance = float(
                np.linalg.norm(
                    point
                    - closest_point
                )
            )

            minimum_distance = min(
                minimum_distance,
                distance,
            )

        return minimum_distance

    def check_collision(
        self,
        position: Sequence[float],
        radius: float = 0.0,
    ) -> bool:
        """
        Проверить столкновение точки/сферы
        с препятствием.

        radius=0:
            обычная точка.

        radius>0:
            приближённая сферическая модель дрона.
        """

        if radius < 0.0:
            raise ValueError(
                "radius не может быть отрицательным."
            )

        distance = (
            self.distance_to_nearest_obstacle(
                position
            )
        )

        return (
            distance
            <= float(radius)
        )

    # =========================================================
    # FLOOR
    # =========================================================

    def _spawn_floor(
        self,
    ) -> None:
        """
        Создать простой floor.

        Используется только если SimulationWorld
        запускается самостоятельно.

        В Base3DEnv вызывается:

            create_floor=False
        """

        size_x = float(
            self.config.size[0]
        )

        size_y = float(
            self.config.size[1]
        )

        half_height = 0.05

        collision_shape = (
            p.createCollisionShape(
                p.GEOM_BOX,

                halfExtents=[
                    size_x / 2.0,
                    size_y / 2.0,
                    half_height,
                ],

                physicsClientId=(
                    self.client_id
                ),
            )
        )

        visual_shape = (
            p.createVisualShape(
                p.GEOM_BOX,

                halfExtents=[
                    size_x / 2.0,
                    size_y / 2.0,
                    half_height,
                ],

                rgbaColor=[
                    0.70,
                    0.70,
                    0.70,
                    1.0,
                ],

                physicsClientId=(
                    self.client_id
                ),
            )
        )

        self.floor_id = (
            p.createMultiBody(
                baseMass=0.0,

                baseCollisionShapeIndex=(
                    collision_shape
                ),

                baseVisualShapeIndex=(
                    visual_shape
                ),

                basePosition=[
                    size_x / 2.0,
                    size_y / 2.0,
                    -half_height,
                ],

                physicsClientId=(
                    self.client_id
                ),
            )
        )

    # =========================================================
    # DYNAMIC OBJECTS
    # =========================================================

    def _generate_dynamic_objects(
        self,
    ) -> List[DynamicObject]:
        """
        Создать динамические объекты.

        Сейчас:

        - car
        - pedestrian

        Машина:
            visual = OBJ CarModel
            collision = один box
            движение = LinearTrajectory

        Пешеход:
            простой cylinder
            движение = WaypointTrajectory
        """

        if not (
            self.config.enable_dynamic_objects
        ):
            return []

        size_x = float(
            self.config.size[0]
        )

        size_y = float(
            self.config.size[1]
        )

        dynamic_objects: List[
            DynamicObject
        ] = []

        # -----------------------------------------------------
        # Cars
        # -----------------------------------------------------

        for index in range(
            self.config.num_cars
        ):
            y = (
                size_y
                * (
                    0.20
                    + 0.12
                    * index
                )
            )

            y = max(
                0.6,
                min(
                    y,
                    size_y - 0.6,
                ),
            )

            car_length = 1.8
            car_width = 0.9
            car_height = 0.8

            car_model = CarModel(
                length=car_length,
                width=car_width,
                height=car_height,
            )

            #
            # Центр box находится на половине
            # высоты машины.
            #
            car_z = (
                car_height
                / 2.0
            )

            car = DynamicObject(
                kind="car",

                geometry="box",

                dimensions=(
                    car_model.bounding_dimensions
                ),

                trajectory=(
                    LinearTrajectory(
                        start=[
                            0.8,
                            y,
                            car_z,
                        ],

                        end=[
                            max(
                                0.8,
                                size_x - 0.8,
                            ),
                            y,
                            car_z,
                        ],

                        speed=1.5,

                        loop=True,
                    )
                ),

                model=car_model,

                color=(
                    0.10,
                    0.25,
                    0.80,
                    1.0,
                ),
            )

            dynamic_objects.append(
                car
            )

        # -----------------------------------------------------
        # Pedestrians
        # -----------------------------------------------------

        for index in range(
            self.config.num_pedestrians
        ):
            x = (
                size_x
                * (
                    0.25
                    + 0.08
                    * index
                )
            )

            x = max(
                0.5,
                min(
                    x,
                    size_x - 0.5,
                ),
            )

            start_y = min(
                0.6,
                size_y / 2.0,
            )

            end_y = max(
                start_y,
                size_y - 0.6,
            )

            pedestrian_height = 1.70

            pedestrian = DynamicObject(
                kind="pedestrian",

                geometry="cylinder",

                dimensions=np.asarray(
                    [
                        0.45,
                        0.45,
                        pedestrian_height,
                    ],
                    dtype=float,
                ),

                trajectory=(
                    WaypointTrajectory(
                        waypoints=[
                            [
                                x,
                                start_y,
                                pedestrian_height
                                / 2.0,
                            ],

                            [
                                x,
                                end_y,
                                pedestrian_height
                                / 2.0,
                            ],
                        ],

                        speed=1.0,

                        loop=True,
                    )
                ),

                color=(
                    0.85,
                    0.45,
                    0.15,
                    1.0,
                ),
            )

            dynamic_objects.append(
                pedestrian
            )

        return dynamic_objects