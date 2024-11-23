import os
import argparse
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.distributions import Normal

# Set random seed for reproducibility
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

# Hyperparameters
LR_ACTOR = 3e-4      # Learning rate for the actor
LR_CRITIC = 3e-4     # Learning rate for the critics
LR_ALPHA = 3e-4      # Learning rate for entropy temperature
GAMMA = 0.99         # Discount factor
TAU = 5e-3           # Soft update parameter
MEMORY_CAPACITY = 1000000
BATCH_SIZE = 256
TARGET_ENTROPY = None  # If None, set to -action_dim
ALPHA = 0.2          # Initial entropy temperature
POLICY_FREQ = 2      # Frequency of delayed policy updates
EVAL_INTERVAL = 50   # Evaluate the policy every 50 episodes
EVAL_EPISODES = 10   # Number of episodes to evaluate

# Experience Replay Buffer using pickle for serialization
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        """Store a transition in the buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample a batch of transitions."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward).reshape(-1,1),
                np.array(next_state), np.array(done).reshape(-1,1))
    
    def __len__(self):
        return len(self.buffer)

    def save(self, path):
        """Save replay buffer to a file using pickle."""
        with open(path, 'wb') as f:
            pickle.dump(self.buffer, f)
    
    def load(self, path):
        """Load replay buffer from a file using pickle."""
        with open(path, 'rb') as f:
            self.buffer = pickle.load(f)
        self.position = len(self.buffer) % self.capacity

# Actor Network (Policy Network)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_dim = action_dim
        self.action_bound = action_bound
        
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
    
    def forward(self, state):
        """Forward pass to compute mean and log_std of action distribution."""
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std
    
    def sample(self, state):
        """Sample an action using the reparameterization trick."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_bound
        
        # Compute log probability
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_bound * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        mean = torch.tanh(mean) * self.action_bound
        return action, log_prob, mean

# Critic Network (Q-Value Network)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.fc1_1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2_1 = nn.Linear(256, 256)
        self.out_1 = nn.Linear(256, 1)
        
        # Q2 architecture
        self.fc1_2 = nn.Linear(state_dim + action_dim, 256)
        self.fc2_2 = nn.Linear(256, 256)
        self.out_2 = nn.Linear(256, 1)
    
    def forward(self, state, action):
        """Forward pass for both critics."""
        xu = torch.cat([state, action], dim=1)
        
        # Q1 forward
        x1 = torch.relu(self.fc1_1(xu))
        x1 = torch.relu(self.fc2_1(x1))
        x1 = self.out_1(x1)
        
        # Q2 forward
        x2 = torch.relu(self.fc1_2(xu))
        x2 = torch.relu(self.fc2_2(x2))
        x2 = self.out_2(x2)
        
        return x1, x2

# SAC Agent
class SAC:
    def __init__(self, state_dim, action_dim, action_bound):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.action_dim = action_dim
        self.action_bound = action_bound
        
        # Initialize networks
        self.actor = Actor(state_dim, action_dim, action_bound).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, action_bound).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        
        # Entropy temperature
        self.alpha = ALPHA
        self.target_entropy = TARGET_ENTROPY if TARGET_ENTROPY else -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR_ALPHA)
        
        # Replay Buffer
        self.memory = ReplayBuffer(MEMORY_CAPACITY)
        
        # Counter for delayed policy updates
        self.total_it = 0
    
    def select_action(self, state, evaluate=False):
        """Select action with or without exploration noise."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if evaluate:
            _, _, action = self.actor.sample(state)
        else:
            action, log_prob, _ = self.actor.sample(state)
        return action.cpu().detach().numpy()[0]
    
    def update_parameters(self):
        """Update actor and critic networks."""
        if len(self.memory) < BATCH_SIZE:
            return
        
        # Increment update counter
        self.total_it += 1
        
        # Sample a batch from memory
        state, action, reward, next_state, done = self.memory.sample(BATCH_SIZE)
        
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)  # Corrected typo here
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        
        with torch.no_grad():
            # Sample actions from the target policy
            next_action, next_log_prob, _ = self.actor_target.sample(next_state)
            # Compute target Q values
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * GAMMA * target_q
        
        # Get current Q estimates
        current_q1, current_q2 = self.critic(state, action)
        
        # Compute critic loss
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
        
        # Optimize Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Delayed policy updates
        if self.total_it % POLICY_FREQ == 0:
            # Sample actions from the current policy
            new_action, log_prob, _ = self.actor.sample(state)
            
            # Obtain both Q1 and Q2 values for the new actions
            critic_q1_new, critic_q2_new = self.critic(state, new_action)
            
            # Take the minimum of Q1 and Q2 to mitigate overestimation
            min_q_new = torch.min(critic_q1_new, critic_q2_new)
            
            # Compute actor loss using the minimum Q-value
            actor_loss = (self.alpha * log_prob - min_q_new).mean()
            
            # Optimize Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Adjust entropy temperature
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
            
            # Soft update target networks
            self.soft_update(self.critic, self.critic_target)
            self.soft_update(self.actor, self.actor_target)
    
    def soft_update(self, source, target):
        """Soft update model parameters."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(TAU * source_param.data + (1.0 - TAU) * target_param.data)
    
    def save(self, checkpoint_path, buffer_path, episode):
        """Save model parameters, optimizers, and replay buffer."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'alpha': self.alpha,
            'log_alpha': self.log_alpha,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            'episode': episode
        }, checkpoint_path)
        # Save replay buffer using pickle
        self.memory.save(buffer_path)
    
    def load(self, checkpoint_path, buffer_path):
        """Load model parameters, optimizers, and replay buffer."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.alpha = checkpoint['alpha']
        self.log_alpha = checkpoint['log_alpha']
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        # Load replay buffer using pickle
        self.memory.load(buffer_path)
        return checkpoint['episode']  # Return the episode number

# Evaluation Function
def evaluate_policy(env, agent, episodes=10):
    """Evaluate the agent's performance without exploration noise."""
    avg_reward = 0.
    for _ in range(episodes):
        state = env.reset(seed=seed)
        if isinstance(state, tuple):
            state = state[0]  # For Gym versions >=0.25
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state, evaluate=True)
            try:
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            except ValueError:
                # For older Gym versions
                next_state, reward, done, info = env.step(action)
            state = next_state
            episode_reward += reward
        avg_reward += episode_reward
    avg_reward /= episodes
    return avg_reward

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Soft Actor-Critic (SAC) for OpenAI Gym Environments')
    parser.add_argument('--env_name', type=str, default='HumanoidStandup-v2', help='Gym environment name')
    parser.add_argument('--train_eps', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--max_steps', type=int, default=1000, help='Max steps per episode')
    parser.add_argument('--load', action='store_true', help='Load trained model')
    parser.add_argument('--save_interval', type=int, default=50, help='Model saving interval')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    args = parser.parse_args()
    
    MAX_EPISODES = args.train_eps
    MAX_STEPS = args.max_steps
    SAVE_INTERVAL = args.save_interval
    
    # Initialize lists to store rewards
    train_rewards = []
    eval_rewards = []
    
    # Create environment
    env = gym.make(args.env_name)
    state = env.reset(seed=seed)
    if isinstance(state, tuple):
        state = state[0]  # For Gym versions >=0.25
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    
    # Create models directory based on environment name
    model_dir = os.path.join('models', args.env_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Initialize SAC Agent
    agent = SAC(state_dim, action_dim, action_bound)
    
    # Optionally load a pre-trained model
    start_episode = 0
    if args.load:
        checkpoint_path = os.path.join(model_dir, 'sac_checkpoint.pth')
        buffer_path = os.path.join(model_dir, 'replay_buffer.pkl')  # Changed to .pkl for pickle
        if os.path.exists(checkpoint_path) and os.path.exists(buffer_path):
            start_episode = agent.load(checkpoint_path, buffer_path)
            print(f"Loaded model from episode {start_episode}")
        else:
            print("Checkpoint or replay buffer not found. Starting from scratch.")
    
    # Training Loop
    for episode in range(start_episode, MAX_EPISODES):
        # Reset the environment for each episode
        state = env.reset(seed=seed)
        if isinstance(state, tuple):
            state = state[0]  # For Gym versions >=0.25
        episode_reward = 0
        done = False
        
        for step in range(MAX_STEPS):
            if args.render:
                env.render()
            
            # Select action with exploration noise
            action = agent.select_action(state, evaluate=False)
            
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
            agent.update_parameters()
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        print(f"Episode: {episode + 1}, Reward: {episode_reward}")
        train_rewards.append(episode_reward)
        
        # Evaluate the policy every EVAL_INTERVAL episodes
        if (episode + 1) % EVAL_INTERVAL == 0:
            eval_reward = evaluate_policy(env, agent, episodes=EVAL_EPISODES)
            print(f"Evaluation over {EVAL_EPISODES} episodes: Average Reward: {eval_reward}")
            eval_rewards.append(eval_reward)
        
        # Save the model and rewards at regular intervals
        if (episode + 1) % SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(model_dir, f'sac_checkpoint_ep{episode + 1}.pth')
            buffer_path = os.path.join(model_dir, f'replay_buffer_ep{episode + 1}.pkl')  # Changed to .pkl for pickle
            agent.save(checkpoint_path, buffer_path, episode + 1)
            print(f"Model saved at episode {episode + 1}")
            
            # Save training and evaluation rewards
            train_rewards_path = os.path.join(model_dir, 'train_rewards.npy')
            eval_rewards_path = os.path.join(model_dir, 'eval_rewards.npy')
            np.save(train_rewards_path, np.array(train_rewards))
            np.save(eval_rewards_path, np.array(eval_rewards))
            print(f"Saved training rewards and evaluation results at episode {episode + 1}")
    
    # After the training loop ends
    final_checkpoint_path = os.path.join(model_dir, f'sac_final_checkpoint.pth')
    final_buffer_path = os.path.join(model_dir, f'replay_buffer_final.pkl')
    agent.save(final_checkpoint_path, final_buffer_path, MAX_EPISODES)
    print("Final model and replay buffer saved.")
    
    # Save final training and evaluation rewards
    final_train_rewards_path = os.path.join(model_dir, 'train_rewards_final.npy')
    final_eval_rewards_path = os.path.join(model_dir, 'eval_rewards_final.npy')
    np.save(final_train_rewards_path, np.array(train_rewards))
    np.save(final_eval_rewards_path, np.array(eval_rewards))
    print("Final training rewards and evaluation results saved.")
    
    env.close()
