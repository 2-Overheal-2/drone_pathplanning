from __future__ import annotations

import argparse
import time

import pybullet as p
import pybullet_data

from world.config import ArenaConfig
from world.world import SimulationWorld


def make_city_config() -> ArenaConfig:
    return ArenaConfig(
        size=(
            30.0,
            15.0,
            12.0,
        ),

        scenario="city",

        density=0.16,

        seed=42,

        start=(
            1.5,
            1.5,
            2.0,
        ),

        goal=(
            28.0,
            13.0,
            3.0,
        ),

        max_objects=150,

        city_block_size=5.0,

        road_width=2.0,

        building_spacing=0.35,

        building_width_range=(
            1.4,
            4.0,
        ),

        building_depth_range=(
            1.4,
            4.0,
        ),

        building_height_range=(
            2.5,
            10.0,
        ),

        column_probability=0.10,

        column_diameter_range=(
            0.30,
            0.70,
        ),

        column_height_range=(
            2.0,
            6.0,
        ),

        enable_power_lines=True,

        num_power_lines=2,

        power_pole_height_range=(
            4.0,
            6.0,
        ),

        power_pole_diameter=0.16,

        power_wire_radius=0.025,

        enable_dynamic_objects=True,

        num_cars=1,

        num_pedestrians=0,
    )


def make_forest_config() -> ArenaConfig:
    return ArenaConfig(
        size=(
            25.0,
            25.0,
            12.0,
        ),

        scenario="forest",

        density=0.10,

        seed=42,

        start=(
            1.5,
            1.5,
            2.0,
        ),

        goal=(
            23.0,
            23.0,
            3.0,
        ),

        max_objects=200,

        tree_diameter_range=(
            0.25,
            0.55,
        ),

        tree_crown_diameter_range=(
            1.2,
            2.6,
        ),

        tree_height_range=(
            3.0,
            9.0,
        ),

        forest_cluster_probability=0.70,

        forest_cluster_radius=3.5,

        forest_cluster_count=5,

        forest_clearing_count=2,

        forest_clearing_radius=2.5,

        enable_dynamic_objects=False,

        num_cars=0,

        num_pedestrians=0,
    )


def print_world_info(
    world: SimulationWorld,
) -> None:
    snapshot = world.snapshot()

    all_objects = (
        snapshot.static_obstacles
        + snapshot.dynamic_obstacles
    )

    kinds = sorted(
        {
            obj["kind"]
            for obj in all_objects
        }
    )

    models = sorted(
        {
            obj["model"]
            for obj in all_objects
            if obj.get("model")
        }
    )

    print()
    print("=== WORLD INFO ===")
    print("Bounds:", snapshot.bounds)
    print("Start:", snapshot.start)
    print("Goal:", snapshot.goal)

    print(
        "Static objects:",
        len(snapshot.static_obstacles),
    )

    print(
        "Dynamic objects:",
        len(snapshot.dynamic_obstacles),
    )

    print(
        "Kinds:",
        kinds,
    )

    print(
        "Models:",
        models,
    )

    print()


def add_debug_markers(
    client_id: int,
    config: ArenaConfig,
) -> None:
    start_visual = (
        p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.20,
            rgbaColor=[
                0.1,
                0.9,
                0.1,
                1.0,
            ],
            physicsClientId=client_id,
        )
    )

    goal_visual = (
        p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.20,
            rgbaColor=[
                0.9,
                0.1,
                0.1,
                1.0,
            ],
            physicsClientId=client_id,
        )
    )

    p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=start_visual,
        basePosition=list(
            config.start
        ),
        physicsClientId=client_id,
    )

    p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=goal_visual,
        basePosition=list(
            config.goal
        ),
        physicsClientId=client_id,
    )


def configure_camera(
    client_id: int,
    config: ArenaConfig,
) -> None:
    size_x = float(
        config.size[0]
    )

    size_y = float(
        config.size[1]
    )

    size_z = float(
        config.size[2]
    )

    camera_distance = (
        max(
            size_x,
            size_y,
        )
        * 0.85
    )

    p.resetDebugVisualizerCamera(
        cameraDistance=camera_distance,

        cameraYaw=45.0,

        cameraPitch=-35.0,

        cameraTargetPosition=[
            size_x / 2.0,
            size_y / 2.0,
            min(
                3.0,
                size_z / 2.0,
            ),
        ],

        physicsClientId=client_id,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "scenario",
        choices=[
            "city",
            "forest",
        ],
    )

    args = parser.parse_args()

    if args.scenario == "city":
        config = make_city_config()

    else:
        config = make_forest_config()

    #
    # Подключение сделано практически так же,
    # как в рабочем car_visual_test.py.
    #
    client_id = p.connect(
        p.GUI,
        options=(
            "--width=1280 "
            "--height=800"
        ),
    )

    if client_id < 0:
        raise RuntimeError(
            "Не удалось открыть PyBullet GUI."
        )

    try:
        print(
            "PyBullet client:",
            client_id,
        )

        #
        # Даём WSLg время физически показать окно.
        #
        time.sleep(
            1.0
        )

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(),
            physicsClientId=client_id,
        )

        p.setGravity(
            0.0,
            0.0,
            -9.81,
            physicsClientId=client_id,
        )

        timestep = (
            1.0
            / 240.0
        )

        p.setTimeStep(
            timestep,
            physicsClientId=client_id,
        )

        print(
            "Generating world..."
        )

        world = SimulationWorld(
            config=config,
            client_id=client_id,
        )

        world.generate()

        print(
            "Spawning world..."
        )

        world.spawn(
            create_floor=True
        )

        print(
            "World spawned."
        )

        add_debug_markers(
            client_id=client_id,
            config=config,
        )

        configure_camera(
            client_id=client_id,
            config=config,
        )

        print_world_info(
            world
        )

        print(
            "Visual scenario:",
            args.scenario,
        )

        print(
            "Green sphere = start"
        )

        print(
            "Red sphere = goal"
        )

        print(
            "Press Ctrl+C in terminal to stop."
        )

        simulation_time = 0.0

        while p.isConnected(
            client_id
        ):
            world.update(
                simulation_time
            )

            p.stepSimulation(
                physicsClientId=client_id,
            )

            simulation_time += (
                timestep
            )

            time.sleep(
                timestep
            )

    except KeyboardInterrupt:
        print()
        print(
            "Visual test stopped."
        )

    finally:
        if p.isConnected(
            client_id
        ):
            p.disconnect(
                client_id
            )


if __name__ == "__main__":
    main()