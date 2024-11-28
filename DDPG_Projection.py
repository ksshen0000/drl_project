#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on [Current Date]

This code implements a constrained DDPG algorithm using projection with PyTorch,
suitable for environments with constraints like HumanoidStandup-v2.
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import gurobipy as gp
from gurobipy import GRB

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# Define the Actor and Critic Networks
# ============================

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_bound = action_bound
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)
        self.tanh = nn.Tanh()

        # Weight initialization
        nn.init.uniform_(self.fc1.weight, -1/np.sqrt(state_dim), 1/np.sqrt(state_dim))
        nn.init.uniform_(self.fc2.weight, -1/np.sqrt(400), 1/np.sqrt(400))
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action = self.tanh(self.fc3(x))
        return action * self.action_bound


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1_s = nn.Linear(state_dim, 400)
        self.fc1_a = nn.Linear(action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

        # Weight initialization
        nn.init.uniform_(self.fc1_s.weight, -1/np.sqrt(state_dim), 1/np.sqrt(state_dim))
        nn.init.uniform_(self.fc1_a.weight, -1/np.sqrt(action_dim), 1/np.sqrt(action_dim))
        nn.init.uniform_(self.fc2.weight, -1/np.sqrt(400), 1/np.sqrt(400))
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)

    def forward(self, state, action):
        x_s = torch.relu(self.fc1_s(state))
        x_a = torch.relu(self.fc1_a(action))
        x = x_s + x_a
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

# ============================
# Replay Buffer
# ============================

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        state = np.array(state)
        action = np.array(action)
        next_state = np.array(next_state)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        return (
            np.array(state_batch),
            np.array(action_batch),
            np.array(reward_batch, dtype=np.float32),
            np.array(next_state_batch),
            np.array(done_batch, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)

# ============================
# Constrained DDPG Agent with Projection
# ============================

class DDPGProjectionAgent:
    def __init__(self, state_dim, action_dim, action_bound, args):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound

        self.gamma = args.gamma
        self.tau = args.tau
        self.lr_actor = args.lr_actor
        self.lr_critic = args.lr_critic
        self.batch_size = args.batch_size
        self.memory_capacity = args.memory_capacity

        self.actor = Actor(state_dim, action_dim, action_bound).to(device)
        self.actor_target = Actor(state_dim, action_dim, action_bound).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        self.memory = ReplayBuffer(self.memory_capacity)
        self.pointer = 0  # To track the number of transitions stored

        # Initialize target networks
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

        # Exploration noise parameters
        self.exploration_noise = args.exploration_noise

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        self.actor.train()
        if not evaluate:
            noise = np.random.normal(0, self.exploration_noise, size=self.action_dim)
            action += noise
        return action.clip(-self.action_bound, self.action_bound)

    def update_parameters(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)

        state_batch = torch.FloatTensor(state_batch).to(device)
        action_batch = torch.FloatTensor(action_batch).to(device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(device)
        done_batch = torch.FloatTensor(np.float32(done_batch)).unsqueeze(1).to(device)

        # Update Critic
        with torch.no_grad():
            next_action_batch = self.actor_target(next_state_batch)
            next_q_values = self.critic_target(next_state_batch, next_action_batch)
            q_target = reward_batch + (1 - done_batch) * self.gamma * next_q_values

        q_values = self.critic(state_batch, action_batch)
        critic_loss = nn.MSELoss()(q_values, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        # For DDPG, we maximize the expected Q value
        actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)

    def projection(self, action, state):
        # Project the action onto the feasible set defined by the constraints
        try:
            with gp.Env(empty=True) as env:
                env.setParam('OutputFlag', 0)
                env.start()
                with gp.Model(env=env) as m:
                    a_vars = m.addVars(self.action_dim, lb=-self.action_bound, ub=self.action_bound, name="a")
                    # Objective: Minimize ||a - action||^2
                    obj = gp.quicksum((a_vars[i] - action[i]) ** 2 for i in range(self.action_dim))
                    m.setObjective(obj, GRB.MINIMIZE)

                    # Add constraints
                    # For example, assume we have a constraint on state[11:11+action_dim] and action
                    w = state[11:11+self.action_dim]
                    constraint_expr = gp.quicksum(a_vars[i] * w[i] for i in range(self.action_dim))
                    m.addConstr(gp.abs_(constraint_expr) <= 20)

                    m.optimize()
                    projected_action = np.array([a_vars[i].X for i in range(self.action_dim)])
            return projected_action
        except gp.GurobiError as e:
            print("Gurobi Error:", e)
            return action  # Return original action if optimization fails

    def hard_update(self, target_net, source_net):
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target_net, source_net):
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def save(self, filepath):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }, filepath)

    @classmethod
    def load(cls, filepath, state_dim, action_dim, action_bound, args):
        agent = cls(state_dim, action_dim, action_bound, args)
        checkpoint = torch.load(filepath)
        agent.actor.load_state_dict(checkpoint['actor'])
        agent.critic.load_state_dict(checkpoint['critic'])
        agent.actor_target.load_state_dict(checkpoint['actor'])
        agent.critic_target.load_state_dict(checkpoint['critic'])
        return agent

# ============================
# Evaluation Function
# ============================

def evaluate_policy(env, agent, episodes=10):
    avg_reward = 0
    for _ in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        done = False
        episode_reward = 0
        while not done:
            action = agent.select_action(state, evaluate=True)
            action = agent.projection(action, state)  # Apply projection
            try:
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            except ValueError:
                next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_reward += reward
        avg_reward += episode_reward
    avg_reward /= episodes
    return avg_reward

# ============================
# Main Function
# ============================

def main():
    parser = argparse.ArgumentParser(description='Constrained DDPG with Projection for OpenAI Gym Environments')

    # Environment and training parameters
    parser.add_argument('--env_name', type=str, default='HumanoidStandup-v2', help='Gym environment name')
    parser.add_argument('--train_eps', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--max_steps', type=int, default=1000, help='Max steps per episode')
    parser.add_argument('--load', action='store_true', help='Load trained model')
    parser.add_argument('--save_interval', type=int, default=50, help='Model saving interval (in episodes)')
    parser.add_argument('--eval_interval', type=int, default=50, help='Evaluation interval (in episodes)')
    parser.add_argument('--eval_episodes', type=int, default=10, help='Number of episodes per evaluation')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')

    # Hyperparameters
    parser.add_argument('--lr_actor', type=float, default=1e-4, help='Learning rate for actor')
    parser.add_argument('--lr_critic', type=float, default=1e-3, help='Learning rate for critic')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.001, help='Soft update parameter')
    parser.add_argument('--memory_capacity', type=int, default=1000000, help='Replay buffer capacity')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--exploration_noise', type=float, default=0.1, help='Exploration noise')

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
    env.seed(args.seed)
    state = env.reset(seed=args.seed)
    if isinstance(state, tuple):
        state = state[0]  # For Gym versions >=0.25

    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = float(env.action_space.high[0])

    # Create models directory based on environment name
    model_dir = os.path.join('models', args.env_name + "_ddpg_projection")
    os.makedirs(model_dir, exist_ok=True)

    # Initialize Agent
    agent = DDPGProjectionAgent(state_dim, action_dim, action_bound, args)

    # Optionally load a pre-trained model
    start_episode = 0
    if args.load:
        checkpoint_path = os.path.join(model_dir, 'ddpg_projection_final_checkpoint.pth')
        if os.path.exists(checkpoint_path):
            agent = DDPGProjectionAgent.load(checkpoint_path, state_dim, action_dim, action_bound, args)
            print(f"Loaded model from {checkpoint_path}")
        else:
            print("Checkpoint not found. Starting from scratch.")

    # Training Loop
    for episode in range(start_episode, start_episode + args.train_eps):
        state = env.reset(seed=args.seed + episode)
        if isinstance(state, tuple):
            state = state[0]  # For Gym versions >=0.25
        episode_reward = 0
        done = False
        step = 0

        while not done:
            step += 1
            if args.render:
                env.render()

            # Select action
            if agent.pointer < 10000:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
                action = (action + np.random.normal(0, 0.1, size=agent.action_dim)).clip(-action_bound, action_bound)

            # Apply projection to satisfy constraints
            action = agent.projection(action, state)

            # Step the environment
            try:
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            except ValueError:
                next_state, reward, done, _ = env.step(action)

            if not isinstance(next_state, np.ndarray):
                next_state = np.array(next_state)

            # Store transition in replay buffer
            done_bool = False if step == args.max_steps else done
            agent.memory.push(state, action, reward, next_state, done_bool)
            agent.pointer += 1

            # Update agent parameters
            if agent.pointer >= 10000:
                agent.update_parameters()

            state = next_state
            episode_reward += reward

            if done or step >= args.max_steps:
                break

        # Append training reward
        train_rewards.append(episode_reward)

        # Exponential moving average of rewards (optional)
        # ewma_reward = 0.05 * episode_reward + (1 - 0.05) * ewma_reward
        # ewma_rewards.append(ewma_reward)

        print(f"Episode: {episode + 1}, Reward: {episode_reward}")

        # Evaluation
        if (episode + 1) % args.eval_interval == 0:
            avg_reward = evaluate_policy(env, agent, episodes=args.eval_episodes)
            eval_rewards.append(avg_reward)
            print(f"Evaluation over {args.eval_episodes} episodes: Average Reward: {avg_reward}")

        # Save the model and rewards at regular intervals
        if (episode + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(model_dir, f'ddpg_projection_checkpoint_ep{episode + 1}.pth')
            agent.save(checkpoint_path)
            print(f"Model saved at episode {episode + 1}")

            # Save training and evaluation rewards
            train_rewards_path = os.path.join(model_dir, 'train_rewards.npy')
            eval_rewards_path = os.path.join(model_dir, 'eval_rewards.npy')
            np.save(train_rewards_path, np.array(train_rewards))
            np.save(eval_rewards_path, np.array(eval_rewards))
            print(f"Saved training rewards and evaluation results at episode {episode + 1}")

        # Check if training is done
        if agent.pointer >= agent.memory_capacity:
            print("Done training")
            break

    # After the training loop ends
    final_checkpoint_path = os.path.join(model_dir, 'ddpg_projection_final_checkpoint.pth')
    agent.save(final_checkpoint_path)
    print("Final model saved.")

    # Save final training and evaluation rewards
    final_train_rewards_path = os.path.join(model_dir, 'train_rewards_final.npy')
    final_eval_rewards_path = os.path.join(model_dir, 'eval_rewards_final.npy')
    np.save(final_train_rewards_path, np.array(train_rewards))
    np.save(final_eval_rewards_path, np.array(eval_rewards))
    print("Final training rewards and evaluation results saved.")

    env.close()

# Entry Point
if __name__ == "__main__":
    main()
