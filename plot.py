import numpy as np
import matplotlib.pyplot as plt
import os

env_name = 'HumanoidStandup-v2'
model_dir_sac = os.path.join('models', env_name+'_sac')
model_dir_sac_mc = os.path.join('models', env_name+'_sac_mc')

# load the rewards from the saved files
sac_rewards = np.load(os.path.join(model_dir_sac, 'train_rewards_final.npy'))
sac_mc_rewards = np.load(os.path.join(model_dir_sac_mc, 'train_rewards_final.npy'))


# plot the training rewards
plt.figure(figsize=(10, 5))
plt.plot(sac_rewards, label='SAC')
plt.plot(sac_mc_rewards, label='SAC-MC')
plt.xlabel('Episodes')
plt.ylabel('Training Rewards')
plt.title(f'Training Rewards for {env_name}')
plt.legend()

plt.show()


