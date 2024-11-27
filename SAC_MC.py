import argparse
import pickle
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from torch.distributions import Normal
import gym
import random

# ============================
# Hyperparameters with Defaults
# ============================

DEFAULT_LR_ACTOR = 3e-4
DEFAULT_LR_CRITIC = 3e-4
DEFAULT_LR_ALPHA = 3e-4
DEFAULT_GAMMA = 0.99
DEFAULT_TAU = 0.005
DEFAULT_ALPHA = 0.2
DEFAULT_TARGET_ENTROPY = None  # If None, set to -action_dim
DEFAULT_MEMORY_CAPACITY = 1000000
DEFAULT_BATCH_SIZE = 256
DEFAULT_POLICY_FREQ = 2
DEFAULT_NUM_EPISODES = 1000
DEFAULT_EVAL_INTERVAL = 50
DEFAULT_EVAL_EPISODES = 10
DEFAULT_SEED = 42

# ============================
# Device Configuration
# ============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# Critic Network
# ============================

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

# ============================
# Actor Network
# ============================

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

# ============================
# Replay Buffer
# ============================

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

# ============================
# SAC Agent
# ============================

class SAC:
    def __init__(self, state_dim, action_dim, action_bound, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.action_dim = action_dim
        self.action_bound = action_bound
        
        # Initialize networks
        self.actor = Actor(state_dim, action_dim, action_bound).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.lr_critic)
        
        # Entropy temperature
        self.alpha = args.alpha
        self.target_entropy = args.target_entropy if args.target_entropy else -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=args.lr_alpha)
        
        # Replay Buffer
        self.memory = ReplayBuffer(args.memory_capacity)
        
        # Hyperparameters
        self.gamma = args.gamma
        self.tau = args.tau
        self.policy_freq = args.policy_freq
        self.batch_size = args.batch_size
        
        # Counter for delayed policy updates
        self.total_it = 0
    
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove unpickleable entries
        state['actor_optimizer'] = None
        state['critic_optimizer'] = None
        state['alpha_optimizer'] = None
        state['memory'] = None
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reinitialize optimizers and memory
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=DEFAULT_LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=DEFAULT_LR_CRITIC)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=DEFAULT_LR_ALPHA)
        self.memory = ReplayBuffer(DEFAULT_MEMORY_CAPACITY)
    
    def select_action(self, state, evaluate=False):
        """Select action with or without exploration noise."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if evaluate:
            with torch.no_grad():
                _, _, action = self.actor.sample(state)
        else:
            with torch.no_grad():
                action, _, _ = self.actor.sample(state)
        return action.cpu().numpy()[0]
    
    def update_parameters(self):
        """Update actor and critic networks."""
        if len(self.memory) < self.batch_size:
            return
        
        # Increment update counter
        self.total_it += 1
        
        # Sample a batch from memory
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        
        with torch.no_grad():
            # Sample actions from the target policy
            next_action, next_log_prob, _ = self.actor_target.sample(next_state)
            # Compute target Q values
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q
        
        # Get current Q estimates
        current_q1, current_q2 = self.critic(state, action)
        
        # Compute critic loss
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
        
        # Optimize Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
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
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)
    
    def save(self, filepath):
        """Save the entire agent using pickle."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath):
        """Load the entire agent using pickle."""
        with open(filepath, 'rb') as f:
            agent = pickle.load(f)
        return agent

# ============================
# Monte Carlo Return Function
# ============================

def monte_carlo_return(rewards, gamma):
    """Compute discounted cumulative returns for a list of rewards."""
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns

# ============================
# Update Critic Using Monte Carlo Returns
# ============================

def update_critic_with_mc(agent, episode_transitions, gamma=DEFAULT_GAMMA):
    """Update Critic networks using Monte Carlo returns."""
    states, actions, rewards, next_states, dones = zip(*episode_transitions)
    returns = monte_carlo_return(rewards, gamma)
    
    states = torch.FloatTensor(states).to(agent.device)
    actions = torch.FloatTensor(actions).to(agent.device)
    returns = torch.FloatTensor(returns).to(agent.device).unsqueeze(1)
    
    # Get current Q estimates
    current_q1, current_q2 = agent.critic(states, actions)
    
    # Compute critic loss using MC returns
    critic_loss = nn.MSELoss()(current_q1, returns) + nn.MSELoss()(current_q2, returns)
    
    # Optimize Critic
    agent.critic_optimizer.zero_grad()
    critic_loss.backward()
    agent.critic_optimizer.step()

# ============================
# Evaluate Policy Function
# ============================

def evaluate_policy(env, agent, episodes=10):
    """Evaluate the current policy over a number of episodes."""
    avg_reward = 0
    for _ in range(episodes):
        state = env.reset(seed=0)
        if isinstance(state, tuple):
            state = state[0]
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

# ============================
# Main Training Loop
# ============================

def main():
    parser = argparse.ArgumentParser(description='Soft Actor-Critic (SAC) for OpenAI Gym Environments')
    # Environment and training parameters
    parser.add_argument('--env_name', type=str, default='HumanoidStandup-v2', help='Gym environment name')
    parser.add_argument('--train_eps', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--max_steps', type=int, default=1000, help='Max steps per episode')
    parser.add_argument('--load', action='store_true', help='Load trained model')
    parser.add_argument('--save_interval', type=int, default=50, help='Model saving interval (in episodes)')
    parser.add_argument('--eval_interval', type=int, default=50, help='Evaluation interval (in episodes)')
    parser.add_argument('--eval_episodes', type=int, default=10, help='Number of episodes per evaluation')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Random seed for reproducibility')
    
    # Hyperparameters for SAC
    parser.add_argument('--lr_actor', type=float, default=DEFAULT_LR_ACTOR, help='Learning rate for actor')
    parser.add_argument('--lr_critic', type=float, default=DEFAULT_LR_CRITIC, help='Learning rate for critic')
    parser.add_argument('--lr_alpha', type=float, default=DEFAULT_LR_ALPHA, help='Learning rate for alpha')
    parser.add_argument('--gamma', type=float, default=DEFAULT_GAMMA, help='Discount factor')
    parser.add_argument('--tau', type=float, default=DEFAULT_TAU, help='Soft update parameter')
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA, help='Initial entropy regularization coefficient')
    parser.add_argument('--target_entropy', type=float, default=DEFAULT_TARGET_ENTROPY, help='Target entropy')
    parser.add_argument('--memory_capacity', type=int, default=DEFAULT_MEMORY_CAPACITY, help='Replay buffer capacity')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size for training')
    parser.add_argument('--policy_freq', type=int, default=DEFAULT_POLICY_FREQ, help='Frequency of policy updates')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Initialize lists to store rewards
    train_rewards = []
    eval_rewards = []
    
    # Create environment
    env = gym.make(args.env_name)
    # env.seed(args.seed)
    state = env.reset(seed=args.seed)
    if isinstance(state, tuple):
        state = state[0]  # For Gym versions >=0.25
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = float(env.action_space.high[0])
    
    # Create models directory based on environment name
    model_dir = os.path.join('models', args.env_name+"_sac_mc")
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize SAC Agent with hyperparameters
    agent = SAC(state_dim, action_dim, action_bound, args)
    
    # Optionally load a pre-trained model
    start_episode = 0
    if args.load:
        checkpoint_path = os.path.join(model_dir, 'sac_checkpoint.pkl')
        if os.path.exists(checkpoint_path):
            agent = SAC.load(checkpoint_path)
            print(f"Loaded model from {checkpoint_path}")
            # Assuming the episode number is stored in the agent or filename; adjust as needed
            # Here, we set start_episode to the last saved episode based on filename
            # For simplicity, set to DEFAULT_NUM_EPISODES // 2
            start_episode = args.train_eps // 2
        else:
            print("Checkpoint not found. Starting from scratch.")
    
    # Training Loop
    for episode in range(start_episode, args.train_eps):
        # Reset the environment for each episode
        state = env.reset(seed=args.seed + episode)
        if isinstance(state, tuple):
            state = state[0]  # For Gym versions >=0.25
        episode_transitions = []
        episode_reward = 0
        done = False

        for step in range(args.max_steps):
            if args.render:
                env.render()
            
            # Select action without exploration noise (deterministic for training)
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
            episode_transitions.append((state, action, reward, next_state, done))
            
            # Update agent
            agent.update_parameters()
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Use Monte Carlo method to update Critic networks
        update_critic_with_mc(agent, episode_transitions, gamma=args.gamma)
        
        # Append training reward
        train_rewards.append(episode_reward)
        
        print(f"Episode: {episode + 1}, Reward: {episode_reward}")
        
        # Evaluation
        if (episode + 1) % args.eval_interval == 0:
            avg_reward = evaluate_policy(env, agent, episodes=args.eval_episodes)
            eval_rewards.append(avg_reward)
            print(f"Evaluation over {args.eval_episodes} episodes: Average Reward: {avg_reward}")
        
        # Save the model and rewards at regular intervals
        if (episode + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(model_dir, f'sac_checkpoint_ep{episode + 1}.pkl')
            agent.save(checkpoint_path)
            print(f"Model saved at episode {episode + 1}")
            
            # Save training and evaluation rewards
            train_rewards_path = os.path.join(model_dir, 'train_rewards.npy')
            eval_rewards_path = os.path.join(model_dir, 'eval_rewards.npy')
            np.save(train_rewards_path, np.array(train_rewards))
            np.save(eval_rewards_path, np.array(eval_rewards))
            print(f"Saved training rewards and evaluation results at episode {episode + 1}")
    
    # After the training loop ends
    final_checkpoint_path = os.path.join(model_dir, 'sac_final_checkpoint.pkl')
    agent.save(final_checkpoint_path)
    print("Final model saved.")
    
    # Save final training and evaluation rewards
    final_train_rewards_path = os.path.join(model_dir, 'train_rewards_final.npy')
    final_eval_rewards_path = os.path.join(model_dir, 'eval_rewards_final.npy')
    np.save(final_train_rewards_path, np.array(train_rewards))
    np.save(final_eval_rewards_path, np.array(eval_rewards))
    print("Final training rewards and evaluation results saved.")
    
    env.close()

# ============================
# Entry Point
# ============================

if __name__ == "__main__":
    main()
