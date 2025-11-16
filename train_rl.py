"""
Training script for RL-based drone defense using Stable-Baselines3.
"""
import os
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from drone_defense_env import DroneDefenseEnv


class CustomCallback(BaseCallback):
    """
    Custom callback for tracking training progress.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Check if episode is done
        if self.locals.get('dones')[0]:
            info = self.locals.get('infos')[0]
            episode_reward = self.locals.get('rewards')[0]

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(info.get('timestep', 0))

            if len(self.episode_rewards) % 10 == 0:
                mean_reward = np.mean(self.episode_rewards[-10:])
                mean_length = np.mean(self.episode_lengths[-10:])
                print(f"\nEpisodes: {len(self.episode_rewards)} | "
                      f"Mean Reward (last 10): {mean_reward:.2f} | "
                      f"Mean Length: {mean_length:.1f}")

        return True


def train_rl_agent(total_timesteps: int = 100000, save_path: str = "models/ppo_drone_defense"):
    """
    Train RL agent using PPO.

    Args:
        total_timesteps: Number of training timesteps
        save_path: Path to save the trained model
    """
    print("=" * 60)
    print("TRAINING RL AGENT FOR DRONE DEFENSE")
    print("=" * 60)
    print(f"\nTraining Configuration:")
    print(f"  Algorithm: PPO (Proximal Policy Optimization)")
    print(f"  Total Timesteps: {total_timesteps:,}")
    print(f"  Environment: Drone Defense with Attention Network")
    print(f"  Model Save Path: {save_path}")
    print("\n" + "=" * 60 + "\n")

    # Create environment
    def make_env():
        return DroneDefenseEnv(
            # map_size=(100, 100),
            # safe_zone=(40, 40, 60, 60),
            # max_attackers=15,
            # max_timesteps=200,
            use_attention=True,  # Train with attention network
            render_mode=None  # No rendering during training
        )

    # Vectorized environment (required by SB3)
    env = DummyVecEnv([make_env])

    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log="./tensorboard_logs/"
    )

    # Create callback
    callback = CustomCallback()

    # Train
    print("Starting training...\n")
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"\nModel saved to {save_path}")

    # Save attention network separately
    attention_network_path = save_path + "_attention_network.pth"
    env_instance = env.envs[0]
    torch.save(env_instance.attention_system.state_dict(), attention_network_path)
    print(f"Attention network saved to {attention_network_path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    return model


def evaluate_agent(model_path: str = "models/ppo_drone_defense", num_episodes: int = 10):
    """
    Evaluate trained agent.

    Args:
        model_path: Path to saved model
        num_episodes: Number of evaluation episodes
    """
    print("\n" + "=" * 60)
    print("EVALUATING TRAINED AGENT")
    print("=" * 60 + "\n")

    # Create environment
    env = DroneDefenseEnv(
        map_size=(100, 100),
        safe_zone=(40, 40, 60, 60),
        max_attackers=15,
        max_timesteps=200,
        use_attention=True,
        render_mode=None
    )

    # Load trained attention network
    attention_network_path = model_path + "_attention_network.pth"
    try:
        env.attention_system.load_state_dict(torch.load(attention_network_path))
        print(f"Loaded attention network from {attention_network_path}\n")
    except Exception as e:
        print(f"Could not load attention network: {e}\n")

    # Load trained model
    model = PPO.load(model_path)

    # Evaluate
    episode_rewards = []
    episode_lengths = []
    victories = 0

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

        episode_rewards.append(total_reward)
        episode_lengths.append(info['timestep'])

        if not info['safe_zone_breached']:
            victories += 1

        print(f"Episode {episode + 1}/{num_episodes} | "
              f"Reward: {total_reward:7.2f} | "
              f"Length: {info['timestep']:3d} | "
              f"Result: {'VICTORY' if not info['safe_zone_breached'] else 'DEFEAT'}")

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Mean Episode Length: {np.mean(episode_lengths):.1f}")
    print(f"  Victory Rate: {victories}/{num_episodes} ({100*victories/num_episodes:.1f}%)")
    print("=" * 60 + "\n")

    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or evaluate RL agent")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"],
                       help="Mode: train or eval")
    parser.add_argument("--timesteps", type=int, default=100000,
                       help="Total training timesteps")
    parser.add_argument("--model-path", type=str, default="models/ppo_drone_defense",
                       help="Path to save/load model")
    parser.add_argument("--eval-episodes", type=int, default=10,
                       help="Number of evaluation episodes")

    args = parser.parse_args()

    if args.mode == "train":
        train_rl_agent(total_timesteps=args.timesteps, save_path=args.model_path)
    else:
        evaluate_agent(model_path=args.model_path, num_episodes=args.eval_episodes)
