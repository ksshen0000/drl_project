import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
def compute_ewma(data, alpha):
    
    ewma = np.zeros_like(data)
    ewma[0] = data[0]

    for t in range(1, len(data)):
        ewma[t] = alpha * data[t] + (1 - alpha) * ewma[t - 1]
    
    return ewma


# 
parser = argparse.ArgumentParser(description='plot the training rewards')
parser.add_argument('--env_name', type=str, default='HumanoidStandup-v2', help='the environment name')
args = parser.parse_args()


env_name = args.env_name
model_dir_sac = os.path.join('models', env_name+'_sac')
model_dir_sac_mc = os.path.join('models', env_name+'_sac_mc')
model_dir_ddpg = os.path.join('models', env_name+'_ddpg')
model_dir_td3 = os.path.join('models', env_name+'_td3')
model_dir_sac_trpo = os.path.join('models', env_name+'_sac_trpo') 
model_dir_ppo = os.path.join('models', env_name+'_ppo')
# load the rewards from the saved files
sac_rewards = np.load(os.path.join(model_dir_sac, 'train_rewards.npy'))
sac_mc_rewards = np.load(os.path.join(model_dir_sac_mc, 'train_rewards.npy'))
td3_rewards = np.load(os.path.join(model_dir_td3, 'train_rewards.npy'))
ddpg_rewards = np.load(os.path.join(model_dir_ddpg, 'train_rewards.npy'))
sac_trpo_rewards = np.load(os.path.join(model_dir_sac_trpo, 'train_rewards.npy'))
ppo_rewards = np.load(os.path.join(model_dir_ppo, 'train_rewards.npy'))
sac_ewma = compute_ewma(sac_rewards, alpha=0.05)
sac_mc_ewma = compute_ewma(sac_mc_rewards, alpha=0.05)
td3_ewma = compute_ewma(td3_rewards, alpha=0.05)
ddpg_ewma = compute_ewma(ddpg_rewards, alpha=0.05)
sac_trpo_ewma = compute_ewma(sac_trpo_rewards, alpha=0.05)
ppo_ewma = compute_ewma(ppo_rewards, alpha=0.05)
# plot the training rewards
plt.figure(figsize=(10, 5))
plt.plot(sac_ewma, label='sac')
plt.plot(sac_mc_ewma, label='sac_mc')
plt.plot(td3_ewma, label='td3')
plt.plot(ddpg_ewma, label='ddpg')
plt.plot(sac_trpo_ewma, label='sac_trpo')
plt.plot(ppo_ewma, label='ppo')
plt.xlabel('Episodes')
plt.ylabel('EWMA training Rewards')
plt.title(f'EWMA training Rewards for {env_name}')
plt.legend()

plt.show()


