from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch as th
import torch.nn as nn
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs import Base3DEnv
from world import ArenaConfig


DEFAULT_CONFIG: Dict[str, Any] = {
    "world": {
        "size": [11.0, 3.0, 6.3],
        "scenario": "city",
        "density": 0.12,
        "seed": 42,
        "start": [1.5, 1.5, 3.0],
        "goal": [9.0, 1.5, 4.5],
        "enable_dynamic_objects": True,
        "num_cars": 0,
        "num_pedestrians": 0,
    },
    "env": {
        "episode_length": 13.0,
        "pyb_freq": 240,
        "ctrl_freq": 30,
        "regenerate_world_on_reset": False,
    },
    "training": {
        "total_timesteps": 5_500_000,
        "n_steps": 1560,
        "batch_size": 195,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "n_epochs": 80,
        "ent_coef": 0.01,
        "vf_coef": 1.0,
        "max_grad_norm": 0.5,
        "normalize_env": True,
        "checkpoint_freq": 50_000,
        "checkpoint_dir": "checkpoints",
        "log_dir": "logs",
        "action_std_init": 0.6,
        "action_std_decay_rate": 0.05,
        "min_action_std": 0.1,
        "action_std_decay_freq": 250_000,
    },
}


class ActionNoiseDecayCallback(BaseCallback):
    """Постепенно уменьшает std Gaussian policy для continuous actions."""

    def __init__(
        self,
        initial_std: float,
        decay_rate: float,
        min_std: float,
        decay_freq: int,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.current_std = float(initial_std)
        self.decay_rate = float(decay_rate)
        self.min_std = float(min_std)
        self.decay_freq = int(decay_freq)

    def _on_step(self) -> bool:
        if (
            self.num_timesteps > 0
            and self.decay_freq > 0
            and self.num_timesteps % self.decay_freq == 0
        ):
            self.current_std = max(
                self.min_std,
                self.current_std - self.decay_rate,
            )

            if hasattr(self.model.policy, "log_std"):
                with th.no_grad():
                    self.model.policy.log_std.fill_(np.log(self.current_std))

            if self.verbose:
                print(
                    f"[PPO] action std -> {self.current_std:.3f} "
                    f"at {self.num_timesteps} timesteps"
                )

        return True


def deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | None) -> dict:
    config = DEFAULT_CONFIG

    if path:
        config_path = Path(path)
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as stream:
                loaded = yaml.safe_load(stream) or {}
            config = deep_merge(DEFAULT_CONFIG, loaded)

    return config


def build_world_config(config: dict) -> ArenaConfig:
    world_cfg = config["world"]

    return ArenaConfig(
        size=tuple(float(x) for x in world_cfg["size"]),
        scenario=str(world_cfg["scenario"]),
        density=float(world_cfg["density"]),
        seed=int(world_cfg["seed"]),
        start=tuple(float(x) for x in world_cfg["start"]),
        goal=tuple(float(x) for x in world_cfg["goal"]),
        enable_dynamic_objects=bool(world_cfg.get("enable_dynamic_objects", True)),
        num_cars=int(world_cfg.get("num_cars", 0)),
        num_pedestrians=int(world_cfg.get("num_pedestrians", 0)),
    )


def make_env(config: dict, gui: bool = False):
    world_config = build_world_config(config)
    env_cfg = config["env"]
    training_cfg = config["training"]

    def _init():
        env = Base3DEnv(
            initial_xyzs=np.asarray([world_config.start], dtype=float),
            gui=gui,
            record=False,
            world_config=world_config,
            pyb_freq=int(env_cfg["pyb_freq"]),
            ctrl_freq=int(env_cfg["ctrl_freq"]),
            episode_length=float(env_cfg["episode_length"]),
            regenerate_world_on_reset=bool(
                env_cfg.get("regenerate_world_on_reset", False)
            ),
        )

        log_dir = Path(training_cfg["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        return Monitor(env, filename=str(log_dir / "monitor.csv"))

    return _init


def train(config_path: str | None, gui: bool = False) -> None:
    config = load_config(config_path)
    training_cfg = config["training"]

    checkpoint_dir = Path(training_cfg["checkpoint_dir"])
    log_dir = Path(training_cfg["log_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("PPO configuration:")
    print(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))

    env = DummyVecEnv([make_env(config, gui=gui)])

    if bool(training_cfg.get("normalize_env", True)):
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=int(training_cfg["checkpoint_freq"]),
        save_path=str(checkpoint_dir),
        name_prefix="uav_ppo",
        save_replay_buffer=False,
        save_vecnormalize=bool(training_cfg.get("normalize_env", True)),
        verbose=1,
    )

    noise_callback = ActionNoiseDecayCallback(
        initial_std=float(training_cfg["action_std_init"]),
        decay_rate=float(training_cfg["action_std_decay_rate"]),
        min_std=float(training_cfg["min_action_std"]),
        decay_freq=int(training_cfg["action_std_decay_freq"]),
        verbose=1,
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=str(log_dir / "tensorboard"),
        learning_rate=float(training_cfg["learning_rate"]),
        n_steps=int(training_cfg["n_steps"]),
        batch_size=int(training_cfg["batch_size"]),
        gamma=float(training_cfg["gamma"]),
        gae_lambda=float(training_cfg["gae_lambda"]),
        clip_range=float(training_cfg["clip_range"]),
        n_epochs=int(training_cfg["n_epochs"]),
        ent_coef=float(training_cfg["ent_coef"]),
        vf_coef=float(training_cfg["vf_coef"]),
        max_grad_norm=float(training_cfg["max_grad_norm"]),
        policy_kwargs={
            "net_arch": dict(pi=[256, 256], vf=[256, 256]),
            "activation_fn": nn.ReLU,
            "ortho_init": True,
        },
        device="cpu",
    )

    if hasattr(model.policy, "log_std"):
        with th.no_grad():
            model.policy.log_std.fill_(
                np.log(float(training_cfg["action_std_init"]))
            )

    try:
        model.learn(
            total_timesteps=int(training_cfg["total_timesteps"]),
            callback=[checkpoint_callback, noise_callback],
            progress_bar=True,
        )

        final_model = checkpoint_dir / "uav_ppo_final"
        model.save(str(final_model))

        if isinstance(env, VecNormalize):
            vec_path = checkpoint_dir / "uav_ppo_final_vecnormalize.pkl"
            env.save(str(vec_path))
            print(f"VecNormalize saved: {vec_path}")

        print(f"Model saved: {final_model}.zip")
    finally:
        env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO in Base3DEnv")
    parser.add_argument(
        "--config",
        default="configs/sb3_racing_config.yaml",
        help="YAML configuration path",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Показывать PyBullet GUI во время обучения (медленно).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.config, gui=args.gui)
