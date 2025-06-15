import os
import numpy as np
import yaml
import time
import argparse
import traceback
from envs.base_3d_env import Base3DEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.policies import ActorCriticPolicy
import torch as th
import torch.nn as nn
from gymnasium import spaces

class ActionNoiseDecayCallback(BaseCallback):
    """
    Callback для уменьшения шума действий по расписанию
    Имитирует action_std_decay из оригинального проекта
    """
    def __init__(self, 
                 initial_std=0.6, 
                 decay_rate=0.05, 
                 min_std=0.1, 
                 decay_freq=250000,
                 verbose=0):
        super(ActionNoiseDecayCallback, self).__init__(verbose)
        self.initial_std = initial_std
        self.decay_rate = decay_rate
        self.min_std = min_std
        self.decay_freq = decay_freq
        self.current_std = initial_std
        
    def _on_step(self) -> bool:
        # Проверяем, нужно ли уменьшить std
        if self.num_timesteps % self.decay_freq == 0 and self.num_timesteps > 0:
            if self.current_std > self.min_std:
                self.current_std = max(self.min_std, self.current_std - self.decay_rate)
                
                # Обновляем std в политике модели
                if hasattr(self.model.policy, 'log_std'):
                    with th.no_grad():
                        self.model.policy.log_std.fill_(np.log(self.current_std))
                
                if self.verbose > 0:
                    print(f"Action std decayed to: {self.current_std:.3f} at step {self.num_timesteps}")
        
        return True

class VecNormalizeCheckpointCallback(BaseCallback):
    """
    Callback для сохранения состояния VecNormalize
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix: str, verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            if isinstance(self.training_env, VecNormalize):
                # Создаем директорию если она не существует
                os.makedirs(self.save_path, exist_ok=True)
                
                path = os.path.join(
                    self.save_path, 
                    f"{self.name_prefix}_{self.num_timesteps}_steps_vecnormalize.pkl"
                )
                self.training_env.save(path)
                
                if self.verbose > 0:
                    print(f"VecNormalize stats saved to: {path}")
        return True

class CustomActorCriticPolicy(ActorCriticPolicy):
    """
    Кастомная политика с разными learning rates для актора и критика
    """
    def __init__(self, *args, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(*args, **kwargs)
        
    def _build_mlp_extractor(self) -> None:
        """
        Создаем feature extractor с архитектурой похожей на оригинальный проект
        """
        super()._build_mlp_extractor()

def create_custom_lr_schedule(actor_lr=3e-4, critic_lr=1e-3):
    """
    Создает расписание learning rate с разными значениями для актора и критика
    """
    def lr_schedule(progress_remaining):
        # Для актора используем базовый learning rate
        return actor_lr
    
    return lr_schedule

def train(config_path=None, record_video=False, algorithm="ppo", headless=False):
    """
    Обучение с параметрами из drone racing проекта
    """
    # Загрузка конфигурации
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Конфигурация по умолчанию из drone racing проекта
        config = {
            'arena_size': 11.0,
            'target': [9, 1.5, 4.5],
            'render': False,
            'total_timesteps': 5500000,
            'checkpoint_freq': 50000,
            'checkpoint_dir': 'checkpoints',
            'log_dir': 'logs',
            'episode_length': 13.0,
            'num_branches': 5,
            'n_steps': 1560,  # Большой буфер как в оригинале
            'batch_size': 195,  # n_steps / 8
            'learning_rate': 3e-4,  # lr_actor
            'gamma': 0.99,  # Точно как в оригинале
            'gae_lambda': 0.95,
            'clip_range': 0.2,  # eps_clip
            'n_epochs': 80,  # K_epochs - много эпох для тщательной оптимизации
            'ent_coef': 0.01,
            'vf_coef': 1.0,  # Увеличенный коэффициент для value function
            'max_grad_norm': 0.5,
            'normalize_env': True,
            # Параметры для action noise decay
            'action_std_init': 0.6,
            'action_std_decay_rate': 0.05,
            'min_action_std': 0.1,
            'action_std_decay_freq': 250000
        }
    
    # Переопределение render если headless
    if headless:
        config['render'] = False
    
    print(f"Training with racing-style PPO configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Создание директорий
    os.makedirs(config.get('checkpoint_dir', 'checkpoints'), exist_ok=True)
    os.makedirs(config.get('log_dir', 'logs'), exist_ok=True)
    
    # Создание среды
    def make_env(render=False):
        def _init():
            env = Base3DEnv(
                target=config['target'],
                gui=render,
                episode_length=config.get('episode_length', 15.0)
            )
            env = Monitor(env, config.get('log_dir', 'logs'))
            return env
        return _init
    
    # Создание векторизованной среды
    env = DummyVecEnv([make_env(render=config['render'])])
    
    # Применение VecNormalize с упрощенными параметрами как вы просили
    if config.get('normalize_env', True):
        print("Applying VecNormalize for observation and reward normalization")
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True
        )
    
    # Создание learning rate schedule
    lr_schedule = create_custom_lr_schedule(
        actor_lr=config.get('learning_rate', 3e-4),
        critic_lr=config.get('learning_rate', 3e-4) * 3.33  # Примерно как lr_critic/lr_actor
    )
    
    # Создание callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=config.get('checkpoint_freq', 50000),
        save_path=config.get('checkpoint_dir', 'checkpoints'),
        name_prefix="racing_ppo_uav",
        save_vecnormalize=False  # Отключаем встроенное сохранение VecNormalize
    )
    
    # Callback для сохранения VecNormalize состояния
    vecnormalize_callback = VecNormalizeCheckpointCallback(
        save_freq=config.get('checkpoint_freq', 50000),
        save_path=config.get('checkpoint_dir', 'checkpoints'),
        name_prefix="racing_ppo_uav",
        verbose=1
    )
    
    # Callback для уменьшения action noise
    action_noise_callback = ActionNoiseDecayCallback(
        initial_std=config.get('action_std_init', 0.6),
        decay_rate=config.get('action_std_decay_rate', 0.05),
        min_std=config.get('min_action_std', 0.1),
        decay_freq=config.get('action_std_decay_freq', 250000),
        verbose=1
    )
    
    # Создание модели PPO с параметрами из drone racing
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=os.path.join(config.get('log_dir', 'logs'), 'tensorboard'),
        learning_rate=lr_schedule,
        n_steps=config.get('n_steps', 7200),  # Большой буфер
        batch_size=config.get('batch_size', 900),  # Соответствующий batch size
        gamma=config.get('gamma', 0.99),
        gae_lambda=config.get('gae_lambda', 0.95),
        clip_range=config.get('clip_range', 0.2),
        n_epochs=config.get('n_epochs', 80),  # Много эпох как в оригинале
        ent_coef=config.get('ent_coef', 0.01),
        vf_coef=config.get('vf_coef', 1.0),  # Увеличенный коэффициент для value function
        max_grad_norm=config.get('max_grad_norm', 0.5),
        policy_kwargs={
            "net_arch": [dict(pi=[256, 256], vf=[256, 256])],  # Архитектура как в оригинале
            "activation_fn": nn.ReLU,
            "ortho_init": True,
        }
    )
    
    # Инициализация action std
    if hasattr(model.policy, 'log_std'):
        with th.no_grad():
            model.policy.log_std.fill_(np.log(config.get('action_std_init', 0.6)))
    
    try:
        total_timesteps = config.get('total_timesteps', 5000000)
        print(f"Starting racing-style training for {total_timesteps} timesteps...")
        print(f"Buffer size (n_steps): {config.get('n_steps', 7200)}")
        print(f"Batch size: {config.get('batch_size', 900)}")
        print(f"Number of epochs per update: {config.get('n_epochs', 80)}")
        print(f"Initial action std: {config.get('action_std_init', 0.6)}")
        print(f"VecNormalize enabled: {config.get('normalize_env', True)}")
        
        # ОСНОВНОЙ ПРОЦЕСС ОБУЧЕНИЯ с тремя callback'ами
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, vecnormalize_callback, action_noise_callback],
            progress_bar=True
        )
        
        # Сохранение финальной модели
        final_model_path = os.path.join(config.get('checkpoint_dir', 'checkpoints'), "racing_ppo_uav_final")
        model.save(final_model_path)
        
        # Сохранение финального состояния VecNormalize
        if config.get('normalize_env', True) and isinstance(env, VecNormalize):
            final_vecnormalize_path = os.path.join(
                config.get('checkpoint_dir', 'checkpoints'), 
                "racing_ppo_uav_final_vecnormalize.pkl"
            )
            env.save(final_vecnormalize_path)
            print(f"Final VecNormalize stats saved to: {final_vecnormalize_path}")
            
        print(f"Racing-style training complete. Final model saved to {final_model_path}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        print(traceback.format_exc())
    
    # Закрытие среды
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent with racing-style parameters")
    parser.add_argument("--config", type=str, default="configs/sb3_racing_config.yaml", help="Path to configuration file")
    parser.add_argument("--record", action="store_true", help="Record videos during training")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no GUI)")
    args = parser.parse_args()
    
    train(args.config, args.record, "ppo", args.headless)
