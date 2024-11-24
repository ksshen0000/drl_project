import os
import argparse
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

# Set random seed for reproducibility
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

# Hyperparameters
LR_A = 1e-4    # Learning rate for actor
LR_C = 1e-3    # Learning rate for critic
GAMMA = 0.99   # Discount factor
TAU = 0.01     # Soft update parameter
MEMORY_CAPACITY = 1000000
BATCH_SIZE = 64
SAVE_INTERVAL = 50  # Save the model every N episodes

# Experience Replay Buffer using pickle
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Store a transition in the buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Sample a batch of transitions"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward).reshape(-1,1),
                np.array(next_state), np.array(done).reshape(-1,1))

    def __len__(self):
        return len(self.buffer)

    def save(self, path):
        """Save replay buffer to a file using pickle"""
        with open(path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load(self, path):
        """Load replay buffer from a file using pickle"""
        with open(path, 'rb') as f:
            self.buffer = pickle.load(f)
        self.position = len(self.buffer) % self.capacity

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.out = nn.Linear(300, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        """Forward pass"""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.out(x))
        # Scale output to [-action_bound, action_bound]
        return x * self.action_bound

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # State pathway
        self.fc1 = nn.Linear(state_dim, 400)
        # Action pathway
        self.fc2 = nn.Linear(action_dim, 300)
        # Combined pathway
        self.fc3 = nn.Linear(400 + 300, 300)
        self.out = nn.Linear(300, 1)

    def forward(self, x, a):
        """Forward pass"""
        xs = torch.relu(self.fc1(x))
        xa = torch.relu(self.fc2(a))
        x = torch.cat([xs, xa], dim=1)
        x = torch.relu(self.fc3(x))
        x = self.out(x)
        return x

# Ornstein-Uhlenbeck Noise for exploration
class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

# DDPG Agent
class DDPG:
    def __init__(self, state_dim, action_dim, action_bound):
        self.actor = Actor(state_dim, action_dim, action_bound).to(self.device())
        self.actor_target = Actor(state_dim, action_dim, action_bound).to(self.device())
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim).to(self.device())
        self.critic_target = Critic(state_dim, action_dim).to(self.device())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.memory = ReplayBuffer(MEMORY_CAPACITY)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_A)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_C)
        self.loss_fn = nn.MSELoss()

        # Initialize Ornstein-Uhlenbeck noise
        self.ou_noise = OUNoise(action_dim)

    def device(self):
        """Return the device to run on (GPU if available, else CPU)"""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def select_action(self, state, add_noise=True):
        """Select action with exploration noise"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device())
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        self.actor.train()
        if add_noise:
            noise = self.ou_noise.noise()
            action += noise
        return np.clip(action, -self.actor.action_bound, self.actor.action_bound)

    def update(self):
        """Update actor and critic networks"""
        if len(self.memory) < BATCH_SIZE:
            return

        state, action, reward, next_state, done = self.memory.sample(BATCH_SIZE)
        state = torch.FloatTensor(state).to(self.device())
        action = torch.FloatTensor(action).to(self.device())
        reward = torch.FloatTensor(reward).to(self.device())
        next_state = torch.FloatTensor(next_state).to(self.device())
        done = torch.FloatTensor(done).to(self.device())

        # Critic update
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = self.critic_target(next_state, next_action)
            y = reward + (1 - done) * GAMMA * target_q
        current_q = self.critic(state, action)
        critic_loss = self.loss_fn(current_q, y)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update of target networks
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

    def soft_update(self, source, target):
        """Soft update model parameters"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(TAU * source_param.data + (1.0 - TAU) * target_param.data)

    def save(self, checkpoint_path, buffer_path, episode):
        """Save model parameters and replay buffer"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'episode': episode
        }, checkpoint_path)
        # Save replay buffer using pickle
        self.memory.save(buffer_path)

    def load(self, checkpoint_path, buffer_path):
        """Load model parameters and replay buffer"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device())
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        # Load replay buffer using pickle
        self.memory.load(buffer_path)
        return checkpoint['episode']  # Return the episode number

def evaluate_policy(env, agent, episodes=5, max_steps=1000):
    """Evaluate the agent's policy without exploration noise."""
    eval_rewards = []
    for ep in range(episodes):
        try:
            state, info = env.reset(seed=seed)
            state = state if isinstance(state, np.ndarray) else np.array(state)
        except TypeError:
            # For older Gym versions
            state = env.reset(seed=seed)
            state = state if isinstance(state, np.ndarray) else np.array(state)

        agent.ou_noise.reset()  # Ensure no noise is added
        episode_reward = 0
        for step in range(max_steps):
            action = agent.select_action(state, add_noise=False)
            try:
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            except ValueError:
                # For older Gym versions
                next_state, reward, done, info = env.step(action)

            if not isinstance(next_state, np.ndarray):
                next_state = np.array(next_state)

            state = next_state
            episode_reward += reward

            if done:
                break
        eval_rewards.append(episode_reward)
    return np.mean(eval_rewards), eval_rewards

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='DDPG for OpenAI Gym Environments')
    parser.add_argument('--env_name', type=str, default='HumanoidStandup-v2', help='Gym environment name')
    parser.add_argument('--train_eps', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--max_steps', type=int, default=1000, help='Max steps per episode')
    parser.add_argument('--load', action='store_true', help='Load trained model')
    parser.add_argument('--save_interval', type=int, default=50, help='Model saving interval')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--eval_interval', type=int, default=50, help='Evaluation interval (in episodes)')
    parser.add_argument('--eval_episodes', type=int, default=5, help='Number of evaluation episodes')
    args = parser.parse_args()

    MAX_EPISODES = args.train_eps
    MAX_STEPS = args.max_steps
    SAVE_INTERVAL = args.save_interval
    EVAL_INTERVAL = args.eval_interval
    EVAL_EPISODES = args.eval_episodes

    # Create environment
    env = gym.make(args.env_name)
    try:
        state, info = env.reset(seed=seed)
        state = state if isinstance(state, np.ndarray) else np.array(state)
    except TypeError:
        # For older Gym versions
        state = env.reset(seed=seed)
        state = state if isinstance(state, np.ndarray) else np.array(state)

    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    # Update the models directory to include environment name
    model_dir = os.path.join('models', args.env_name+"_ddpg")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Initialize Agent
    agent = DDPG(state_dim, action_dim, action_bound)

    # Initialize Ornstein-Uhlenbeck noise
    agent.ou_noise.reset()

    # Optionally load model
    start_episode = 0
    if args.load:
        # Specify the paths to the saved model and replay buffer
        checkpoint_path = os.path.join(model_dir, 'ddpg_checkpoint.pth')
        buffer_path = os.path.join(model_dir, 'replay_buffer.pkl')  # Changed to .pkl for pickle
        if os.path.exists(checkpoint_path) and os.path.exists(buffer_path):
            start_episode = agent.load(checkpoint_path, buffer_path)
            print(f"Loaded model from episode {start_episode}")
        else:
            print("Checkpoint or replay buffer not found. Starting from scratch.")

    # Initialize lists to store rewards
    train_rewards = []
    eval_rewards_list = []

    # Training Loop
    for episode in range(start_episode, MAX_EPISODES):
        # Reset the environment for each episode
        try:
            state, info = env.reset(seed=seed)
            state = state if isinstance(state, np.ndarray) else np.array(state)
        except TypeError:
            # For older Gym versions
            state = env.reset(seed=seed)
            state = state if isinstance(state, np.ndarray) else np.array(state)

        # Reset noise at the start of each episode
        agent.ou_noise.reset()

        episode_reward = 0
        for step in range(MAX_STEPS):
            if args.render:
                env.render()

            # Select action with exploration noise
            action = agent.select_action(state, add_noise=True)

            # Step the environment
            try:
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            except ValueError:
                # For older Gym versions
                next_state, reward, done, info = env.step(action)

            # Convert next_state to numpy array if it's not
            if not isinstance(next_state, np.ndarray):
                next_state = np.array(next_state)

            # Store transition in replay buffer
            agent.memory.push(state, action, reward, next_state, done)

            # Update agent
            agent.update()

            state = next_state
            episode_reward += reward

            if done:
                break

        train_rewards.append(episode_reward)
        print(f"Episode: {episode + 1}, Reward: {episode_reward}")

        # Save the model at regular intervals
        if (episode + 1) % SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(model_dir, f'ddpg_checkpoint_ep{episode + 1}.pth')
            buffer_path = os.path.join(model_dir, f'replay_buffer_ep{episode + 1}.pkl')  # Changed to .pkl
            agent.save(checkpoint_path, buffer_path, episode + 1)
            print(f"Model saved at episode {episode + 1}")

        # Perform evaluation at specified intervals
        if (episode + 1) % EVAL_INTERVAL == 0:
            eval_mean_reward, eval_rewards = evaluate_policy(env, agent, episodes=EVAL_EPISODES, max_steps=MAX_STEPS)
            eval_rewards_list.append(eval_mean_reward)
            print(f"Evaluation after Episode {episode + 1}: Mean Reward: {eval_mean_reward}")

            # Optionally, save evaluation results immediately
            eval_save_path = os.path.join(model_dir, 'eval_rewards.npy')
            np.save(eval_save_path, np.array(eval_rewards_list))
            train_save_path = os.path.join(model_dir, 'train_rewards.npy')
            np.save(train_save_path, np.array(train_rewards))
            print("Saved training and evaluation rewards.")

    # After training, save the final rewards
    np.save(os.path.join(model_dir, 'train_rewards.npy'), np.array(train_rewards))
    np.save(os.path.join(model_dir, 'eval_rewards.npy'), np.array(eval_rewards_list))
    print("Training complete. Rewards saved.")

    env.close()
