from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import ActionType, DroneModel, ObservationType, Physics

from world import ArenaConfig, SimulationWorld


class Base3DEnv(BaseRLAviary):
    """
    Gymnasium/RL-адаптер над SimulationWorld.

    SimulationWorld отвечает за сцену.
    BaseRLAviary отвечает за физику дрона.
    Этот класс связывает их и определяет reward/termination/truncation.
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
        if initial_xyzs is None:
            initial_xyzs = np.asarray([[1.5, 1.5, 3.0]], dtype=float)
        else:
            initial_xyzs = np.asarray(initial_xyzs, dtype=float).reshape(1, 3)

        target_point = tuple(
            float(x) for x in (
                target if target is not None else (9.0, 1.5, 4.5)
            )
        )

        if world_config is None:
            if arena_size is None:
                size = (11.0, 3.0, 6.3)
            elif isinstance(arena_size, (int, float)):
                # Backward compatibility со старым arena_size=11.
                size = (float(arena_size), 3.0, 6.3)
            else:
                if len(arena_size) != 3:
                    raise ValueError("arena_size должен быть числом или последовательностью [x, y, z].")
                size = tuple(float(x) for x in arena_size)

            world_config = ArenaConfig(
                size=size,
                scenario=scenario,
                density=float(density),
                seed=int(seed),
                start=tuple(float(x) for x in initial_xyzs[0]),
                goal=target_point,
                enable_dynamic_objects=bool(enable_dynamic_objects),
                num_cars=int(num_cars),
                num_pedestrians=int(num_pedestrians),
            )

        world_config.validate()

        self.world_config = world_config
        self.world: Optional[SimulationWorld] = None
        self.regenerate_world_on_reset = bool(regenerate_world_on_reset)
        self._world_reset_index = 0

        self.EPISODE_LEN_SEC = float(episode_length)
        self.target = np.asarray(world_config.goal, dtype=np.float32)
        self.TARGET_POS = self.target.copy()

        self.arena_length = float(world_config.size[0])
        self.arena_width = float(world_config.size[1])
        self.arena_height = float(world_config.size[2])
        self.dangerous_height = 0.10
        self.boundary_margin = 0.30
        self.drone_collision_radius = 0.10

        self.obstacle_data = []
        self.target_id = None
        self.prev_drone_pos = initial_xyzs[0].copy()

        # Статистика.
        self.target_reached_count = 0
        self.collision_count = 0
        self.episode_count = 0
        self.out_of_bounds_count = 0
        self.timeout_count = 0
        self.tilted_count = 0

        # BaseRLAviary сам вызывает _addObstacles(), но у нас он переопределён как pass.
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

        self._setup_world(regenerate=True)
        self.prev_drone_pos = self._getDroneStateVector(0)[:3].copy()

        if self.GUI:
            self._setup_target_marker()
            self._setup_camera()

    def _addObstacles(self):
        """
        BaseRLAviary вызывает этот hook внутри своего housekeeping.
        Мир создаём отдельно в _setup_world(), чтобы не смешивать ответственности.
        """
        pass

    def _setup_world(self, regenerate: bool) -> None:
        if regenerate or self.world is None:
            config = self.world_config

            if self.regenerate_world_on_reset and self._world_reset_index > 0:
                config = config.with_seed(self.world_config.seed + self._world_reset_index)

            self.world = SimulationWorld(config=config, client_id=self.CLIENT).generate()

        # BaseAviary уже создаёт plane.urdf, поэтому второй пол здесь не создаём.
        self.world.spawn(create_floor=False)
        self._sync_obstacle_data()

    def _sync_obstacle_data(self) -> None:
        if self.world is None:
            self.obstacle_data = []
            return

        snapshot = self.world.snapshot()
        self.obstacle_data = snapshot.static_obstacles + snapshot.dynamic_obstacles

    def get_environment_snapshot(self):
        if self.world is None:
            raise RuntimeError("SimulationWorld ещё не создан.")
        return self.world.snapshot()

    def step(self, action):
        # step_counter измеряется в PyBullet ticks.
        if self.world is not None:
            simulation_time = self.step_counter / float(self.PYB_FREQ)
            self.world.update(simulation_time)
            self._sync_obstacle_data()

        obs, reward, terminated, truncated, info = super().step(action)

        if self.GUI:
            self._setup_camera()

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.episode_count += 1
        obs, info = super().reset(seed=seed, options=options)

        self._world_reset_index += 1
        self._setup_world(regenerate=self.regenerate_world_on_reset)

        self.prev_drone_pos = self._getDroneStateVector(0)[:3].copy()

        if self.GUI:
            self._setup_target_marker()
            self._setup_camera()

        # Observation зависит только от состояния дрона, но возвращаем свежий info.
        obs = self._computeObs()
        info = self._computeInfo()
        return obs, info

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        drone_pos = state[0:3]
        angular_velocity = state[13:16]

        current_distance = float(np.linalg.norm(self.target - drone_pos))
        previous_distance = float(np.linalg.norm(self.target - self.prev_drone_pos))

        # Основной shaping: награда за прогресс к цели.
        reward = 4.5 * (previous_distance - current_distance)
        reward -= 0.001 * float(np.linalg.norm(angular_velocity))

        # Небольшой штраф при приближении к препятствиям.
        min_obstacle_distance = self._minimum_obstacle_distance(drone_pos)
        if min_obstacle_distance < 0.40:
            reward -= 0.05 * (0.40 - min_obstacle_distance)

        # Терминальные события не перезаписываются последующим else.
        if self._target_reached(drone_pos):
            reward += 20.0
        elif self._has_collision(drone_pos):
            reward -= 10.0
        elif self._out_of_bounds(drone_pos) or self._too_tilted(state):
            reward -= 10.0

        self.prev_drone_pos = drone_pos.copy()
        return float(reward)

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        drone_pos = state[0:3]

        if self._target_reached(drone_pos):
            self.target_reached_count += 1
            return True

        if self._has_collision(drone_pos):
            self.collision_count += 1
            return True

        return False

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        drone_pos = state[0:3]

        if self._out_of_bounds(drone_pos):
            self.out_of_bounds_count += 1
            return True

        if self.step_counter / float(self.PYB_FREQ) >= self.EPISODE_LEN_SEC:
            self.timeout_count += 1
            return True

        if self._too_tilted(state):
            self.tilted_count += 1
            return True

        return False

    def _computeInfo(self):
        return {
            "target_position": self.target.tolist(),
            "target_reached_count": self.target_reached_count,
            "collision_count": self.collision_count,
            "out_of_bounds_count": self.out_of_bounds_count,
            "timeout_count": self.timeout_count,
            "tilted_count": self.tilted_count,
            "episode_count": self.episode_count,
            "scenario": self.world_config.scenario,
            "arena_size": list(self.world_config.size),
        }

    def get_stats(self):
        return {
            **self._computeInfo(),
            "success_rate": self.target_reached_count / max(1, self.episode_count),
        }

    def _target_reached(self, drone_pos: np.ndarray) -> bool:
        # Геометрически нормальная цель вместо старого "x > target.x".
        return float(np.linalg.norm(self.target - drone_pos)) <= 0.45

    def _out_of_bounds(self, drone_pos: np.ndarray) -> bool:
        x, y, z = drone_pos
        return not (
            0.0 <= x <= self.arena_length
            and 0.0 <= y <= self.arena_width
            and self.dangerous_height <= z <= self.arena_height
        )

    @staticmethod
    def _too_tilted(state: np.ndarray) -> bool:
        # state[7:10] = roll, pitch, yaw.
        return abs(float(state[7])) > 0.6 or abs(float(state[8])) > 0.6

    def _has_collision(self, drone_pos: np.ndarray) -> bool:
        return self._minimum_obstacle_distance(drone_pos) < self.drone_collision_radius

    def _minimum_obstacle_distance(self, drone_pos: np.ndarray) -> float:
        if not self.obstacle_data:
            return float("inf")

        minimum = float("inf")

        for obstacle in self.obstacle_data:
            aabb_min = np.asarray(obstacle["aabb_min"], dtype=float)
            aabb_max = np.asarray(obstacle["aabb_max"], dtype=float)
            closest = np.minimum(np.maximum(drone_pos, aabb_min), aabb_max)
            distance = float(np.linalg.norm(drone_pos - closest))
            minimum = min(minimum, distance)

        return minimum

    def _setup_target_marker(self) -> None:
        visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.25,
            rgbaColor=[1.0, 0.1, 0.1, 0.8],
            physicsClientId=self.CLIENT,
        )
        self.target_id = p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=visual,
            basePosition=self.target.tolist(),
            physicsClientId=self.CLIENT,
        )

    def _setup_camera(self) -> None:
        center = [
            self.arena_length / 2.0,
            self.arena_width / 2.0,
            min(self.arena_height / 2.0, 5.0),
        ]
        distance = max(3.0, min(35.0, max(self.arena_length, self.arena_width) * 0.8))

        p.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=45.0,
            cameraPitch=-35.0,
            cameraTargetPosition=center,
            physicsClientId=self.CLIENT,
        )
