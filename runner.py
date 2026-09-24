from envs.base_3d_env import Base3DEnv

from algorithms.adapter import AlgorithmAdapter
from algorithms.policies.dummy import DummyPolicyController

from planners.direct import DirectPlanner

from simulation.runner import SimulationRunner


def main():
    print(
        "=== PATH PLANNER TEST ==="
    )

    env = Base3DEnv(
        gui=False,
        regenerate_world_on_reset=True,
    )

    try:
        planner_adapter = AlgorithmAdapter(
            DirectPlanner()
        )

        planner_runner = SimulationRunner(
            env=env,
            algorithm_adapter=planner_adapter,
        )

        planner_runner.reset()

        result = (
            planner_runner.plan_path()
        )

        print(
            "Algorithm:",
            result.metadata.get(
                "algorithm"
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

    finally:
        env.close()

    print()
    print(
        "=== POLICY TEST ==="
    )

    env = Base3DEnv(
        gui=False,
        regenerate_world_on_reset=True,
    )

    try:
        policy_adapter = AlgorithmAdapter(
            DummyPolicyController()
        )

        policy_runner = SimulationRunner(
            env=env,
            algorithm_adapter=policy_adapter,
        )

        observation, info = (
            policy_runner.reset()
        )

        print(
            "Initial observation:",
            observation.shape,
        )

        result = (
            policy_runner.policy_step()
        )

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = result

        print(
            "Observation:",
            observation.shape,
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

    finally:
        env.close()

    print()
    print(
        "SimulationRunner test: OK"
    )


if __name__ == "__main__":
    main()