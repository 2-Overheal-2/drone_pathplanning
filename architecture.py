from envs.base_3d_env import Base3DEnv

from algorithms.adapter import AlgorithmAdapter
from planners.direct import DirectPlanner


def main():
    env = Base3DEnv(
        gui=False,
        regenerate_world_on_reset=True,
    )

    try:
        print("=== INITIAL ENVIRONMENT ===")

        print(
            "Observation space:",
            env.observation_space,
        )

        print(
            "Action space:",
            env.action_space,
        )

        snapshot = (
            env.get_environment_snapshot()
        )

        print(
            "Static obstacles:",
            len(
                snapshot.static_obstacles
            ),
        )

        print(
            "Dynamic obstacles:",
            len(
                snapshot.dynamic_obstacles
            ),
        )

        print(
            "Start:",
            snapshot.start,
        )

        print(
            "Goal:",
            snapshot.goal,
        )

        print(
            "Bounds:",
            snapshot.bounds,
        )

        for obstacle in (
            snapshot.static_obstacles
            + snapshot.dynamic_obstacles
        ):
            assert (
                "body_id"
                not in obstacle
            )

        print(
            "Snapshot isolation: OK"
        )

        adapter = AlgorithmAdapter(
            DirectPlanner()
        )

        result = adapter.plan(
            env.world
        )

        print()
        print(
            "=== PLANNER ==="
        )

        print(
            "Algorithm:",
            result.metadata.get(
                "algorithm"
            ),
        )

        print(
            "Type:",
            result.metadata.get(
                "algorithm_type"
            ),
        )

        print(
            "Success:",
            result.success,
        )

        print(
            "Path:",
            result.path,
        )

        print(
            "Path length:",
            result.metadata.get(
                "path_length"
            ),
        )

        print()
        print(
            "=== RESET TEST ==="
        )

        for index in range(3):
            obs, info = env.reset()

            snapshot = (
                env.get_environment_snapshot()
            )

            print(
                f"Reset {index + 1}:",
                f"static={len(snapshot.static_obstacles)},",
                f"dynamic={len(snapshot.dynamic_obstacles)},",
                f"obs_shape={obs.shape}",
            )

        print()
        print(
            "=== STEP TEST ==="
        )

        action = (
            env.action_space.sample()
            * 0.0
        )

        result = env.step(
            action
        )

        obs, reward, terminated, truncated, info = (
            result
        )

        print(
            "Observation shape:",
            obs.shape,
        )

        print(
            "Reward:",
            reward,
        )

        print(
            "Terminated:",
            terminated,
        )

        print(
            "Truncated:",
            truncated,
        )

        print()
        print(
            "Architecture smoke test: OK"
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()