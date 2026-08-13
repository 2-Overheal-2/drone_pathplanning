from __future__ import annotations

import argparse
import time

import pybullet as p

from planners import DirectPlanner, PlannerAdapter
from world import ArenaConfig, SimulationWorld


def add_marker(client_id: int, position, color, radius: float = 0.25):
    visual = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=color,
        physicsClientId=client_id,
    )
    return p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=visual,
        basePosition=list(position),
        physicsClientId=client_id,
    )


def draw_path(client_id: int, path):
    for a, b in zip(path[:-1], path[1:]):
        p.addUserDebugLine(
            a,
            b,
            lineColorRGB=[1.0, 0.0, 0.0],
            lineWidth=3.0,
            lifeTime=0.0,
            physicsClientId=client_id,
        )


def main():
    parser = argparse.ArgumentParser(description="Standalone PyBullet arena demo")
    parser.add_argument("--scenario", choices=["city", "forest", "empty"], default="city")
    parser.add_argument("--size", nargs=3, type=float, default=[30.0, 30.0, 15.0])
    parser.add_argument("--density", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cars", type=int, default=1)
    parser.add_argument("--pedestrians", type=int, default=1)
    parser.add_argument(
        "--direct-path",
        action="store_true",
        help="Показать тестовый прямой путь start -> goal.",
    )
    args = parser.parse_args()

    sx, sy, sz = args.size
    start = (max(0.5, sx * 0.05), max(0.5, sy * 0.05), min(1.5, sz * 0.25))
    goal = (sx * 0.90, sy * 0.90, min(max(2.0, sz * 0.30), sz * 0.85))

    config = ArenaConfig(
        size=(sx, sy, sz),
        scenario=args.scenario,
        density=args.density,
        seed=args.seed,
        start=start,
        goal=goal,
        enable_dynamic_objects=(args.cars + args.pedestrians) > 0,
        num_cars=args.cars,
        num_pedestrians=args.pedestrians,
    )

    client_id = p.connect(p.GUI)
    if client_id < 0:
        raise RuntimeError("Не удалось подключиться к PyBullet GUI.")

    p.setGravity(0.0, 0.0, -9.81, physicsClientId=client_id)
    p.setTimeStep(1.0 / 240.0, physicsClientId=client_id)

    world = SimulationWorld(config=config, client_id=client_id).generate()
    world.spawn(create_floor=True)

    add_marker(client_id, config.start, [0.1, 0.9, 0.1, 0.9])
    add_marker(client_id, config.goal, [0.9, 0.1, 0.1, 0.9])

    snapshot = world.snapshot()
    print(f"Scenario: {config.scenario}")
    print(f"Arena: {config.size}")
    print(f"Static obstacles: {len(snapshot.static_obstacles)}")
    print(f"Dynamic obstacles: {len(snapshot.dynamic_obstacles)}")

    if args.direct_path:
        adapter = PlannerAdapter(DirectPlanner())
        result = adapter.run(world)
        print("Planner:", result.metadata)
        draw_path(client_id, result.path)

    p.resetDebugVisualizerCamera(
        cameraDistance=max(5.0, min(40.0, max(sx, sy) * 0.9)),
        cameraYaw=45.0,
        cameraPitch=-40.0,
        cameraTargetPosition=[sx / 2.0, sy / 2.0, min(sz / 3.0, 5.0)],
        physicsClientId=client_id,
    )

    simulation_time = 0.0
    dt = 1.0 / 240.0

    try:
        while p.isConnected(client_id):
            world.update(simulation_time)
            p.stepSimulation(physicsClientId=client_id)
            time.sleep(dt)
            simulation_time += dt
    except KeyboardInterrupt:
        pass
    finally:
        if p.isConnected(client_id):
            p.disconnect(physicsClientId=client_id)


if __name__ == "__main__":
    main()
