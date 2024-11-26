import argparse
import os
import gym
import numpy as np
import torch
from torch import nn
from torch.optim import Adam

# Define the PPO Agent
class PPO:
    def __init__(self, state_dim, action_dim, action_bound, args):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound

        self.gamma = 0.99
        self.epsilon = 0.2
        self.lr = 3e-4
        self.epochs = 10
        self.batch_size = 64

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim, action_bound).to(self.device)
        self.critic = Critic(state_dim).to(self.device)

        self.optimizer_actor = Adam(self.actor.parameters(), lr=self.lr)
        self.optimizer_critic = Adam(self.critic.parameters(), lr=self.lr)

        self.memory = []

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action_mean, action_std = self.actor(state)
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        action = action.cpu().detach().numpy()
        action = np.clip(action, -self.action_bound, self.action_bound)
        log_prob = dist.log_prob(torch.FloatTensor(action).to(self.device)).sum()
        return action, log_prob.item()

    def store_transition(self, transition):
        self.memory.append(transition)

    def update(self):
        state_arr = torch.FloatTensor([t[0] for t in self.memory]).to(self.device)
        action_arr = torch.FloatTensor([t[1] for t in self.memory]).to(self.device)
        logprob_arr = torch.FloatTensor([t[2] for t in self.memory]).to(self.device)
        reward_arr = [t[3] for t in self.memory]
        done_arr = [t[4] for t in self.memory]

        # Compute discounted rewards
        returns = []
        Gt = 0
        for reward, done in zip(reversed(reward_arr), reversed(done_arr)):
            if done:
                Gt = 0
            Gt = reward + self.gamma * Gt
            returns.insert(0, Gt)
        returns = torch.FloatTensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Update policy and value network
        for _ in range(self.epochs):
            for index in range(0, len(self.memory), self.batch_size):
                batch_slice = slice(index, index + self.batch_size)
                states = state_arr[batch_slice]
                actions = action_arr[batch_slice]
                old_logprobs = logprob_arr[batch_slice]
                returns_batch = returns[batch_slice]

                action_mean, action_std = self.actor(states)
                dist = torch.distributions.Normal(action_mean, action_std)
                logprobs = dist.log_prob(actions).sum(1)
                state_values = self.critic(states).squeeze()

                ratios = torch.exp(logprobs - old_logprobs)
                advantages = returns_batch - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = nn.MSELoss()(state_values, returns_batch)

                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                self.optimizer_actor.step()

                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                self.optimizer_critic.step()

        self.memory = []

    def save(self, checkpoint_path, episode):
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'episode': episode
        }
        torch.save(checkpoint, checkpoint_path)

    def load(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            episode = checkpoint['episode']
            print(f"Loaded model starting from episode {episode}")
            return episode
        else:
            print("Checkpoint not found, starting from scratch.")
            return 0

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_bound = action_bound
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, x):
        x = self.fc(x)
        action_mean = self.mean(x)
        action_log_std = self.log_std(x)
        action_std = torch.exp(action_log_std)
        return action_mean, action_std

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        value = self.fc(x)
        return value

def evaluate_policy(env, agent, episodes=5):
    eval_rewards = []
    for _ in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        done = False
        total_reward = 0
        while not done:
            action, _ = agent.select_action(state)
            try:
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            except ValueError:
                next_state, reward, done, info = env.step(action)
            state = next_state
            total_reward += reward
        eval_rewards.append(total_reward)
    return np.mean(eval_rewards)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Proximal Policy Optimization (PPO) for OpenAI Gym Environments')
    parser.add_argument('--env_name', type=str, default='HumanoidStandup-v2', help='Gym environment name')
    parser.add_argument('--train_eps', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--max_steps', type=int, default=500, help='Max steps per episode')
    parser.add_argument('--load', action='store_true', help='Load trained model')
    parser.add_argument('--save_interval', type=int, default=50, help='Model saving interval')
    parser.add_argument('--eval_interval', type=int, default=10, help='Model evaluation interval')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    args = parser.parse_args()

    MAX_EPISODES = args.train_eps
    MAX_STEPS = args.max_steps
    SAVE_INTERVAL = args.save_interval
    EVAL_INTERVAL = args.eval_interval
    EVAL_EPISODES = 5

    # Initialize lists to store rewards
    train_rewards = []
    eval_rewards = []

    # Create environment
    env = gym.make(args.env_name)
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]  # For Gym versions >=0.25

    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    # Create models directory
    model_dir = os.path.join('models', args.env_name + '_ppo')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Initialize PPO Agent
    agent = PPO(state_dim, action_dim, action_bound, args)

    # Optionally load a pre-trained model
    start_episode = 0
    checkpoint_path = os.path.join(model_dir, 'ppo_checkpoint.pth')
    if args.load:
        start_episode = agent.load(checkpoint_path)

    # Training loop
    for episode in range(start_episode, MAX_EPISODES):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        episode_reward = 0
        done = False

        for step in range(MAX_STEPS):
            if args.render:
                env.render()

            action, log_prob = agent.select_action(state)

            try:
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            except ValueError:
                next_state, reward, done, info = env.step(action)

            if not isinstance(next_state, np.ndarray):
                next_state = np.array(next_state)

            agent.store_transition((state, action, log_prob, reward, done))

            state = next_state
            episode_reward += reward

            if done:
                break

        agent.update()
        print(f"Episode: {episode + 1}, Reward: {episode_reward}")
        train_rewards.append(episode_reward)

        # Evaluate the policy every EVAL_INTERVAL episodes
        if (episode + 1) % EVAL_INTERVAL == 0:
            eval_reward = evaluate_policy(env, agent, episodes=EVAL_EPISODES)
            print(f"Evaluation over {EVAL_EPISODES} episodes: Average Reward: {eval_reward}")
            eval_rewards.append(eval_reward)

        # Save the model and rewards at regular intervals
        if (episode + 1) % SAVE_INTERVAL == 0:
            agent.save(checkpoint_path, episode + 1)
            print(f"Model saved at episode {episode + 1}")

            # Save training and evaluation rewards
            train_rewards_path = os.path.join(model_dir, 'train_rewards.npy')
            eval_rewards_path = os.path.join(model_dir, 'eval_rewards.npy')
            np.save(train_rewards_path, np.array(train_rewards))
            np.save(eval_rewards_path, np.array(eval_rewards))
            print(f"Training and evaluation rewards saved at episode {episode + 1}")

    # Save the final model after training ends
    final_checkpoint_path = os.path.join(model_dir, 'ppo_final_checkpoint.pth')
    agent.save(final_checkpoint_path, MAX_EPISODES)
    print("Final model saved.")

    # Save the final training and evaluation rewards
    final_train_rewards_path = os.path.join(model_dir, 'train_rewards_final.npy')
    final_eval_rewards_path = os.path.join(model_dir, 'eval_rewards_final.npy')
    np.save(final_train_rewards_path, np.array(train_rewards))
    np.save(final_eval_rewards_path, np.array(eval_rewards))
    print("Final training and evaluation rewards saved.")

    env.close()
 