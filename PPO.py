import argparse
import os
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class SaveRewardsCallback(BaseCallback):
    """
    Custom callback for saving training and evaluation rewards.
    """
    def __init__(self, eval_env, eval_freq, eval_episodes, save_path, verbose=0):
        super(SaveRewardsCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.save_path = save_path
        self.train_rewards = []
        self.eval_rewards = []
        self.best_mean_reward = -np.inf

    def _on_step(self):
        # Save training reward
        if len(self.locals['infos']) > 0 and 'episode' in self.locals['infos'][0]:
            ep_info = self.locals['infos'][0]['episode']
            ep_reward = ep_info['r']
            self.train_rewards.append(ep_reward)

        # Evaluate the policy every eval_freq steps
        if self.n_calls % self.eval_freq == 0:
            eval_rewards = []
            for _ in range(self.eval_episodes):
                obs, info = self.eval_env.reset()
                done = False
                total_reward = 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                eval_rewards.append(total_reward)
            mean_eval_reward = np.mean(eval_rewards)
            self.eval_rewards.append(mean_eval_reward)
            print(f"Evaluation at step {self.n_calls}: Mean Reward = {mean_eval_reward}")

            # Save the best model
            if mean_eval_reward > self.best_mean_reward:
                self.best_mean_reward = mean_eval_reward
                best_model_path = os.path.join(self.save_path, 'best_model')
                self.model.save(best_model_path)
                print(f"New best model saved at step {self.n_calls} with mean reward {mean_eval_reward}")

            # Save rewards
            np.save(os.path.join(self.save_path, 'train_rewards.npy'), np.array(self.train_rewards))
            np.save(os.path.join(self.save_path, 'eval_rewards.npy'), np.array(self.eval_rewards))
            print(f"Training and evaluation rewards saved at step {self.n_calls}")

        return True

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train PPO agent with Stable Baselines3')
    parser.add_argument('--env_name', type=str, default='HumanoidStandup-v2', help='Gym environment name')
    parser.add_argument('--total_timesteps', type=int, default=1e6, help='Total timesteps for training')
    parser.add_argument('--eval_freq', type=int, default=10000, help='Evaluation frequency (timesteps)')
    parser.add_argument('--eval_episodes', type=int, default=5, help='Number of episodes for evaluation')
    parser.add_argument('--save_path', type=str, default='./models/', help='Directory to save models and rewards')
    parser.add_argument('--load', type=str, default=None, help='Path to a pre-trained model')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()

    # Create models directory
    model_dir = os.path.join(args.save_path, args.env_name + '_ppo')
    os.makedirs(model_dir, exist_ok=True)

    # Create environment
    env = gym.make(args.env_name)
    env.reset(seed=args.seed)

    # Create evaluation environment
    eval_env = gym.make(args.env_name)
    eval_env.reset(seed=args.seed + 100)

    # Define the model
    model = PPO('MlpPolicy', env, verbose=1, seed=args.seed)

    # Optionally load a pre-trained model
    if args.load is not None:
        model = PPO.load(args.load, env=env)
        print(f"Loaded model from {args.load}")

    # Define the custom callback
    eval_callback = SaveRewardsCallback(
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        save_path=model_dir,
        verbose=1
    )

    # Train the agent
    model.learn(total_timesteps=int(args.total_timesteps), callback=eval_callback)

    # Save the final model
    final_model_path = os.path.join(model_dir, 'ppo_final_model')
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Save the final training and evaluation rewards
    np.save(os.path.join(model_dir, 'train_rewards_final.npy'), np.array(eval_callback.train_rewards))
    np.save(os.path.join(model_dir, 'eval_rewards_final.npy'), np.array(eval_callback.eval_rewards))
    print("Final training and evaluation rewards saved.")

    # Close environments
    env.close()
    eval_env.close()
