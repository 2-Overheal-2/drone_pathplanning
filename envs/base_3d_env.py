from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence

import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import (
    ActionType,
    DroneModel,
    ObservationType,
    Physics,
)

from world.config import ArenaConfig
from world.world import EnvironmentSnapshot, SimulationWorld


class Base3DEnv(BaseRLAviary):
    """
    Gymnasium/RL-адаптер над SimulationWorld.

    Ответственность модулей:

    SimulationWorld:
        - генерация сцены;
        - статические препятствия;
        - динамические препятствия;
        - обновление динамических объектов;
        - EnvironmentSnapshot;
        - collision queries.

    BaseRLAviary:
        - физика БПЛА;
        - observation space;
        - action space;
        - обработка RPM-команд.

    Base3DEnv:
        - связывает SimulationWorld и BaseRLAviary;
        - reward;
        - terminated;
        - truncated;
        - статистика эпизодов.
    """

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        initial_xyzs: Optional[np.ndarray] = None,
        physics: Physics = Physics.PYB,
        pyb_freq: int = 240,
        ctrl_freq: int = 30,
        gui: bool = False,
        record: bool = False,
        target: Optional[Sequence[float]] = None,
        arena_size: Optional[Sequence[float]] = None,
        scenario: str = "city",
        density: float = 0.12,
        seed: int = 42,
        episode_length: float = 13.0,
        enable_dynamic_objects: bool = True,
        num_cars: int = 0,
        num_pedestrians: int = 0,
        world_config: Optional[ArenaConfig] = None,
        regenerate_world_on_reset: bool = False,
    ):
        """
        Создать RL-среду.

        Можно настроить мир двумя способами:

        1. Через отдельные параметры:
            arena_size
            scenario
            density
            seed
            target
            ...

        2. Передать готовый ArenaConfig через world_config.

        Если передан world_config, он является основной
        конфигурацией мира.
        """

        if world_config is None:
            if initial_xyzs is None:
                initial_xyzs = np.asarray(
                    [
                        [
                            1.5,
                            1.5,
                            3.0,
                        ]
                    ],
                    dtype=float,
                )
            else:
                initial_xyzs = np.asarray(
                    initial_xyzs,
                    dtype=float,
                ).reshape(1, 3)

            if target is None:
                target_point = (
                    9.0,
                    1.5,
                    4.5,
                )
            else:
                target_point = tuple(
                    float(value)
                    for value in target
                )

            if arena_size is None:
                size = (
                    11.0,
                    3.0,
                    6.3,
                )

            elif isinstance(
                arena_size,
                (int, float),
            ):
                # Совместимость со старым API:
                # arena_size=11
                size = (
                    float(arena_size),
                    3.0,
                    6.3,
                )

            else:
                if len(arena_size) != 3:
                    raise ValueError(
                        "arena_size должен быть числом "
                        "или последовательностью [x, y, z]."
                    )

                size = tuple(
                    float(value)
                    for value in arena_size
                )

            world_config = ArenaConfig(
                size=size,
                scenario=scenario,
                density=float(density),
                seed=int(seed),
                start=tuple(
                    float(value)
                    for value in initial_xyzs[0]
                ),
                goal=target_point,
                enable_dynamic_objects=bool(
                    enable_dynamic_objects
                ),
                num_cars=int(num_cars),
                num_pedestrians=int(
                    num_pedestrians
                ),
            )

        else:
            world_config.validate()

            #
            # Если world_config передан,
            # его start используется как положение дрона
            # по умолчанию.
            #
            if initial_xyzs is None:
                initial_xyzs = np.asarray(
                    [
                        world_config.start
                    ],
                    dtype=float,
                )

            else:
                initial_xyzs = np.asarray(
                    initial_xyzs,
                    dtype=float,
                ).reshape(1, 3)

                #
                # Явно переданное initial_xyzs
                # имеет приоритет над start в config.
                #
                world_config = replace(
                    world_config,
                    start=tuple(
                        float(value)
                        for value
                        in initial_xyzs[0]
                    ),
                )

            #
            # Явно переданный target также
            # имеет приоритет.
            #
            if target is not None:
                world_config = replace(
                    world_config,
                    goal=tuple(
                        float(value)
                        for value in target
                    ),
                )

        world_config.validate()

        #
        # Основная конфигурация мира.
        #
        self.world_config = world_config

        #
        # Сохраняем исходный seed отдельно,
        # чтобы при регенерации получать:
        #
        # base_seed + 1
        # base_seed + 2
        # base_seed + 3
        #
        # а не случайно накапливать его.
        #
        self._base_world_seed = int(
            world_config.seed
        )

        #
        # SimulationWorld появится после
        # инициализации BaseRLAviary.
        #
        self.world: Optional[
            SimulationWorld
        ] = None

        self.regenerate_world_on_reset = bool(
            regenerate_world_on_reset
        )

        self._world_reset_index = 0

        #
        # Параметры эпизода.
        #
        self.EPISODE_LEN_SEC = float(
            episode_length
        )

        self.target = np.asarray(
            world_config.goal,
            dtype=np.float32,
        )

        #
        # Оставляем старое имя для совместимости
        # с существующим кодом.
        #
        self.TARGET_POS = (
            self.target.copy()
        )

        #
        # Размеры арены.
        #
        self.arena_length = float(
            world_config.size[0]
        )

        self.arena_width = float(
            world_config.size[1]
        )

        self.arena_height = float(
            world_config.size[2]
        )

        self.dangerous_height = 0.10

        self.boundary_margin = 0.30

        #
        # Пока дрон для collision-query
        # аппроксимируется сферой.
        #
        self.drone_collision_radius = 0.10

        #
        # GUI target marker.
        #
        self.target_id = None

        #
        # Нужен для reward за прогресс.
        #
        self.prev_drone_pos = (
            initial_xyzs[0].copy()
        )

        #
        # Общая статистика.
        #
        self.target_reached_count = 0
        self.collision_count = 0
        self.episode_count = 0
        self.out_of_bounds_count = 0
        self.timeout_count = 0
        self.tilted_count = 0

        #
        # BaseRLAviary отвечает за физику дрона.
        #
        # Observation/action space оставляем теми же,
        # что были в исходной PPO-среде:
        #
        # ObservationType.KIN
        # ActionType.RPM
        #
        super().__init__(
            drone_model=drone_model,
            num_drones=1,
            initial_xyzs=initial_xyzs,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=ObservationType.KIN,
            act=ActionType.RPM,
        )

        #
        # BaseRLAviary уже создал:
        #
        # plane
        # drone
        #
        # Теперь добавляем наш SimulationWorld.
        #
        self._setup_world(
            regenerate=True,
            pybullet_was_reset=False,
        )

        self.prev_drone_pos = (
            self._getDroneStateVector(
                0
            )[:3].copy()
        )

        if self.GUI:
            self._setup_target_marker()
            self._setup_camera()

    def _addObstacles(
        self,
    ) -> None:
        """
        Hook BaseRLAviary.

        BaseRLAviary вызывает этот метод во время
        собственной инициализации/reset.

        Здесь намеренно ничего не создаём.

        Все препятствия принадлежат SimulationWorld
        и создаются через _setup_world().
        """

        pass

    def _setup_world(
        self,
        regenerate: bool = False,
        pybullet_was_reset: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        """
        Создать или восстановить SimulationWorld.

        regenerate:
            заново сгенерировать расположение объектов.

        pybullet_was_reset:
            True после BaseRLAviary.reset(), потому что
            родитель уже вызвал resetSimulation().

        seed:
            новый seed для генерации мира.
        """

        if seed is not None:
            self.world_config = replace(
                self.world_config,
                seed=int(seed),
            )

        #
        # Первый запуск.
        #
        if self.world is None:
            self.world = SimulationWorld(
                config=self.world_config,
                client_id=self.CLIENT,
            )

            self.world.generate()

            #
            # BaseRLAviary уже создаёт plane.urdf.
            #
            self.world.spawn(
                create_floor=False
            )

            return

        #
        # Если config изменился,
        # передаём его существующему миру.
        #
        self.world.config = (
            self.world_config
        )

        self.world.reset(
            regenerate=regenerate,
            create_floor=False,
            pybullet_was_reset=(
                pybullet_was_reset
            ),
        )

    def get_environment_snapshot(
        self,
    ) -> EnvironmentSnapshot:
        """
        Получить нейтральное описание среды.

        Используется внешними алгоритмами:
        A*
        RRT
        PRM
        и другими adapter-модулями.
        """

        if self.world is None:
            raise RuntimeError(
                "SimulationWorld ещё не создан."
            )

        return self.world.snapshot()

    def step(
        self,
        action,
    ):
        """
        Выполнить один управляющий шаг.

        Перед физическим шагом дрона обновляем
        кинематические динамические препятствия.
        """

        if self.world is not None:
            #
            # step_counter хранится в PyBullet ticks.
            #
            simulation_time = (
                self.step_counter
                / float(self.PYB_FREQ)
            )

            self.world.update(
                simulation_time
            )

        #
        # BaseRLAviary:
        #
        # action -> RPM -> physics
        # -> observation/reward/termination/info
        #
        obs, reward, terminated, truncated, info = (
            super().step(
                action
            )
        )

        if self.GUI:
            self._setup_camera()

        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )

    def reset(
        self,
        seed=None,
        options=None,
    ):
        """
        Начать новый эпизод.

        Сначала BaseRLAviary полностью пересоздаёт
        PyBullet simulation.

        После этого заново создаём объекты
        SimulationWorld.
        """

        self.episode_count += 1

        #
        # BaseRLAviary.reset() вызывает
        # PyBullet resetSimulation().
        #
        obs, info = super().reset(
            seed=seed,
            options=options,
        )

        #
        # Старый target_id уже уничтожен
        # resetSimulation().
        #
        self.target_id = None

        self._world_reset_index += 1

        regenerate = bool(
            self.regenerate_world_on_reset
        )

        world_seed = None

        if regenerate:
            #
            # Если Gymnasium передал seed явно,
            # используем его и для генератора мира.
            #
            if seed is not None:
                world_seed = int(seed)

            else:
                world_seed = (
                    self._base_world_seed
                    + self._world_reset_index
                )

        #
        # Важно:
        #
        # pybullet_was_reset=True говорит
        # SimulationWorld не удалять старые body_id,
        # потому что PyBullet уже уничтожил их.
        #
        self._setup_world(
            regenerate=regenerate,
            pybullet_was_reset=True,
            seed=world_seed,
        )

        self.prev_drone_pos = (
            self._getDroneStateVector(
                0
            )[:3].copy()
        )

        if self.GUI:
            self._setup_target_marker()
            self._setup_camera()

        #
        # Observation space не изменён,
        # поэтому просто получаем свежий KIN observation.
        #
        obs = self._computeObs()

        info = self._computeInfo()

        return (
            obs,
            info,
        )

    def _computeReward(
        self,
    ) -> float:
        """
        Reward сохраняем совместимым
        с предыдущей версией среды.
        """

        state = (
            self._getDroneStateVector(
                0
            )
        )

        drone_pos = state[
            0:3
        ]

        angular_velocity = state[
            13:16
        ]

        current_distance = float(
            np.linalg.norm(
                self.target
                - drone_pos
            )
        )

        previous_distance = float(
            np.linalg.norm(
                self.target
                - self.prev_drone_pos
            )
        )

        #
        # Награда за прогресс к цели.
        #
        reward = (
            4.5
            * (
                previous_distance
                - current_distance
            )
        )

        #
        # Небольшой штраф за угловую скорость.
        #
        reward -= (
            0.001
            * float(
                np.linalg.norm(
                    angular_velocity
                )
            )
        )

        #
        # Штраф при приближении к препятствию.
        #
        min_obstacle_distance = (
            self._minimum_obstacle_distance(
                drone_pos
            )
        )

        if (
            min_obstacle_distance
            < 0.40
        ):
            reward -= (
                0.05
                * (
                    0.40
                    - min_obstacle_distance
                )
            )

        #
        # Терминальные события.
        #
        if self._target_reached(
            drone_pos
        ):
            reward += 20.0

        elif self._has_collision(
            drone_pos
        ):
            reward -= 10.0

        elif (
            self._out_of_bounds(
                drone_pos
            )
            or self._too_tilted(
                state
            )
        ):
            reward -= 10.0

        self.prev_drone_pos = (
            drone_pos.copy()
        )

        return float(
            reward
        )

    def _computeTerminated(
        self,
    ) -> bool:
        """
        Настоящее завершение задачи:

        - достигнута цель;
        - столкновение.
        """

        state = (
            self._getDroneStateVector(
                0
            )
        )

        drone_pos = state[
            0:3
        ]

        if self._target_reached(
            drone_pos
        ):
            self.target_reached_count += 1
            return True

        if self._has_collision(
            drone_pos
        ):
            self.collision_count += 1
            return True

        return False

    def _computeTruncated(
        self,
    ) -> bool:
        """
        Принудительное завершение эпизода:

        - выход за границы;
        - timeout;
        - слишком большой наклон.
        """

        state = (
            self._getDroneStateVector(
                0
            )
        )

        drone_pos = state[
            0:3
        ]

        if self._out_of_bounds(
            drone_pos
        ):
            self.out_of_bounds_count += 1
            return True

        simulation_time = (
            self.step_counter
            / float(self.PYB_FREQ)
        )

        if (
            simulation_time
            >= self.EPISODE_LEN_SEC
        ):
            self.timeout_count += 1
            return True

        if self._too_tilted(
            state
        ):
            self.tilted_count += 1
            return True

        return False

    def _computeInfo(
        self,
    ) -> dict:
        """
        Дополнительная информация среды.
        """

        return {
            "target_position": (
                self.target.tolist()
            ),

            "target_reached_count": (
                self.target_reached_count
            ),

            "collision_count": (
                self.collision_count
            ),

            "out_of_bounds_count": (
                self.out_of_bounds_count
            ),

            "timeout_count": (
                self.timeout_count
            ),

            "tilted_count": (
                self.tilted_count
            ),

            "episode_count": (
                self.episode_count
            ),

            "scenario": (
                self.world_config.scenario
            ),

            "arena_size": list(
                self.world_config.size
            ),
        }

    def get_stats(
        self,
    ) -> dict:
        """
        Общая статистика всех эпизодов.
        """

        return {
            **self._computeInfo(),

            "success_rate": (
                self.target_reached_count
                / max(
                    1,
                    self.episode_count,
                )
            ),
        }

    def _target_reached(
        self,
        drone_pos: np.ndarray,
    ) -> bool:
        """
        Проверить достижение цели.
        """

        distance = float(
            np.linalg.norm(
                self.target
                - drone_pos
            )
        )

        return (
            distance
            <= 0.45
        )

    def _out_of_bounds(
        self,
        drone_pos: np.ndarray,
    ) -> bool:
        """
        Проверить выход за границы арены.
        """

        x, y, z = (
            drone_pos
        )

        return not (
            0.0
            <= x
            <= self.arena_length

            and 0.0
            <= y
            <= self.arena_width

            and self.dangerous_height
            <= z
            <= self.arena_height
        )

    @staticmethod
    def _too_tilted(
        state: np.ndarray,
    ) -> bool:
        """
        state[7:10]:
            roll
            pitch
            yaw
        """

        roll = float(
            state[7]
        )

        pitch = float(
            state[8]
        )

        return (
            abs(roll) > 0.6
            or abs(pitch) > 0.6
        )

    def _has_collision(
        self,
        drone_pos: np.ndarray,
    ) -> bool:
        """
        Проверить столкновение с объектом мира.

        Base3DEnv больше не знает,
        какие конкретно obstacle types существуют.
        """

        if self.world is None:
            return False

        return self.world.check_collision(
            position=drone_pos,
            radius=(
                self.drone_collision_radius
            ),
        )

    def _minimum_obstacle_distance(
        self,
        drone_pos: np.ndarray,
    ) -> float:
        """
        Расстояние до ближайшего препятствия.

        Вычислением занимается SimulationWorld.
        """

        if self.world is None:
            return float(
                "inf"
            )

        return (
            self.world
            .distance_to_nearest_obstacle(
                drone_pos
            )
        )

    def _setup_target_marker(
        self,
    ) -> None:
        """
        Создать визуальный маркер цели.

        Только GUI.
        """

        visual_shape = (
            p.createVisualShape(
                p.GEOM_SPHERE,

                radius=0.25,

                rgbaColor=[
                    1.0,
                    0.1,
                    0.1,
                    0.8,
                ],

                physicsClientId=(
                    self.CLIENT
                ),
            )
        )

        self.target_id = (
            p.createMultiBody(
                baseMass=0.0,

                baseVisualShapeIndex=(
                    visual_shape
                ),

                basePosition=(
                    self.target.tolist()
                ),

                physicsClientId=(
                    self.CLIENT
                ),
            )
        )

    def _setup_camera(
        self,
    ) -> None:
        """
        Настроить обзор GUI.
        """

        center = [
            self.arena_length
            / 2.0,

            self.arena_width
            / 2.0,

            min(
                self.arena_height
                / 2.0,
                5.0,
            ),
        ]

        distance = max(
            3.0,

            min(
                35.0,

                max(
                    self.arena_length,
                    self.arena_width,
                )
                * 0.8,
            ),
        )

        p.resetDebugVisualizerCamera(
            cameraDistance=distance,

            cameraYaw=45.0,

            cameraPitch=-35.0,

            cameraTargetPosition=center,

            physicsClientId=(
                self.CLIENT
            ),
        )