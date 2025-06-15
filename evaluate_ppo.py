import os
import numpy as np
import yaml
import time
import argparse
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.base_3d_env import Base3DEnv
import traceback
from stable_baselines3.common.monitor import Monitor

def evaluate_racing_model(model_path, config_path=None, num_episodes=100, render=True, render_delay=0.05, record_video=False):
    """
    Оценка модели, обученной с racing-style параметрами
    """
    # Загрузка конфигурации
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'arena_size': 11.0,
            'target': [9, 1.5, 4.5],
            'render': True,
            'episode_length': 13.0
        }
    
    print("Evaluating racing-style PPO model with configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Создание среды
    def make_env():
        def _init():
            env = Base3DEnv(
                target=config['target'],
                gui=config['render'],
                episode_length=config.get('episode_length', 13.0)
            )
            # env = Monitor(env, 'logs')
            return env
        return _init
    
    env = DummyVecEnv([make_env()])
    
    # Загрузка статистики нормализации
    try:
        env = VecNormalize.load(f"{os.path.splitext(model_path)[0]}_vecnormalize.pkl", env)
        env.training = False
        env.norm_reward = False
        print("Loaded normalization statistics")
    except FileNotFoundError:
        print("No normalization statistics found, using raw environment")
    
    # Загрузка модели
    model = PPO.load(model_path)
    print(f"Loaded racing-style PPO model from {model_path}")
    
    # Цикл оценки
    all_rewards = []
        
    obs = env.reset()
    done = False
    for i in range(100):
        done = False
        total_reward = 0.0
        while not done:
            try:
                    # Получение действия от модели (детерминированно для оценки)
                action, _ = model.predict(obs, deterministic=True)
                    
                    # Выполнение действия
                next_obs, reward, done, info = env.step(action)
                total_reward += float(reward)
                all_rewards.append(reward[0])
                obs = next_obs
                           
                if done:
                    env.reset()
                        
                        # Определение исхода эпизода           
            except Exception as e:
                print(f"Error during evaluation step: {e}")
                print(traceback.format_exc())
                break



    
    # Добавляем скользящее среднее для лучшей визуализации
    # window_size = min(10, len(all_rewards) // 5)
    # if window_size > 1:
    #     moving_avg = np.convolve(all_rewards, np.ones(window_size)/window_size, mode='valid')
    #     plt.plot(range(window_size-1, len(all_rewards)), moving_avg, 
    #             color='red', linewidth=2, label=f'Moving Average (window={window_size})')
    #     plt.legend()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate racing-style PPO agent")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model file (.zip)")
    parser.add_argument("--config", type=str, default="configs/sb3_racing_config.yaml", help="Path to configuration file")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to evaluate")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--delay", type=float, default=0.05, help="Render delay for smoother visualization")
    parser.add_argument("--record", action="store_true", help="Record videos of evaluation episodes")
    args = parser.parse_args()
    
    evaluate_racing_model(args.model, args.config, args.episodes, args.render, args.delay, args.record)
