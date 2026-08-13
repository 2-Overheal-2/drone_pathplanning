from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ppo_train import build_world_config, load_config
from envs import Base3DEnv


def make_eval_env(config: dict, render: bool, record: bool):
    world_config = build_world_config(config)
    env_cfg = config["env"]

    def _init():
        return Base3DEnv(
            initial_xyzs=np.asarray([world_config.start], dtype=float),
            gui=render,
            record=record,
            world_config=world_config,
            pyb_freq=int(env_cfg["pyb_freq"]),
            ctrl_freq=int(env_cfg["ctrl_freq"]),
            episode_length=float(env_cfg["episode_length"]),
            regenerate_world_on_reset=False,
        )

    return _init


def infer_vecnormalize_path(model_path: str) -> Path:
    model = Path(model_path)
    stem = model.stem if model.suffix == ".zip" else model.name
    return model.with_name(f"{stem}_vecnormalize.pkl")


def evaluate(
    model_path: str,
    config_path: str | None,
    episodes: int,
    render: bool,
    delay: float,
    record: bool,
    vecnormalize_path: str | None = None,
) -> None:
    config = load_config(config_path)

    env = DummyVecEnv([make_eval_env(config, render=render, record=record)])

    normalize = bool(config["training"].get("normalize_env", True))
    if normalize:
        candidate = (
            Path(vecnormalize_path)
            if vecnormalize_path
            else infer_vecnormalize_path(model_path)
        )

        if candidate.exists():
            env = VecNormalize.load(str(candidate), env)
            env.training = False
            env.norm_reward = False
            print(f"Loaded VecNormalize: {candidate}")
        else:
            print(
                f"WARNING: VecNormalize expected but not found: {candidate}. "
                "Evaluation will use raw observations."
            )

    model = PPO.load(model_path, env=env, device="cpu")

    rewards = []
    successes = 0
    previous_target_counter = 0

    try:
        for episode in range(1, episodes + 1):
            obs = env.reset()
            done = False
            total_reward = 0.0
            final_info = {}

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, dones, infos = env.step(action)

                total_reward += float(reward[0])
                done = bool(dones[0])
                final_info = infos[0]

                if render and delay > 0:
                    time.sleep(delay)

            rewards.append(total_reward)

            target_counter = int(final_info.get("target_reached_count", 0))
            episode_success = target_counter > previous_target_counter
            previous_target_counter = target_counter
            successes += int(episode_success)

            print(
                f"Episode {episode}/{episodes}: "
                f"reward={total_reward:.3f}, "
                f"success={episode_success}"
            )

        mean_reward = float(np.mean(rewards)) if rewards else 0.0
        std_reward = float(np.std(rewards)) if rewards else 0.0

        print()
        print(f"Episodes: {episodes}")
        print(f"Mean reward: {mean_reward:.3f}")
        print(f"Std reward: {std_reward:.3f}")
        print(f"Successes: {successes}/{episodes}")
    finally:
        env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPO model")
    parser.add_argument("--model", required=True, help="Path to .zip model")
    parser.add_argument(
        "--config",
        default="configs/sb3_racing_config.yaml",
        help="YAML configuration path",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--record", action="store_true")
    parser.add_argument(
        "--vecnormalize",
        default=None,
        help="Optional explicit path to VecNormalize .pkl",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        model_path=args.model,
        config_path=args.config,
        episodes=args.episodes,
        render=args.render,
        delay=args.delay,
        record=args.record,
        vecnormalize_path=args.vecnormalize,
    )
