import numpy as np 
import os 
from os import environ
import sys 
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)
from ray import tune
# import ray
# N_THREADS = '1'
# environ['OMP_NUM_THREADS'] = N_THREADS
# environ['OPENBLAS_NUM_THREADS'] = N_THREADS
# environ['MKL_NUM_THREADS'] = N_THREADS
# environ['VECLIB_MAXIMUM_THREADS'] = N_THREADS
# environ['NUMEXPR_NUM_THREADS'] = N_THREADS
from simulation.all_markets_simulation import Market, config
from ray.rllib.algorithms.ppo import PPOConfig
import pandas as pd 

from ray.rllib.evaluate import run 
import seaborn as sns






env_config = config.copy()

# restore best agent 
path = f'{parent_dir}/ray_results/40_noise'
analysis = tune.ExperimentAnalysis(path, default_metric="episode_reward_mean", default_mode="max")
best_trial = analysis.get_best_trial(metric="episode_reward_mean", mode="max", scope="last")
config = analysis.get_best_config()
path = analysis.get_best_checkpoint(trial=best_trial, mode="max", metric="episode_reward_mean")
print(f'best checkpoint: {path}')
config['num_workers'] = 1
# config['num_workers'] = 1
AC = PPOConfig()
AC.update_from_dict(config)
agent = AC.build()
agent.restore(path)    
eval_config = config['evaluation_config']['env_config']

M = Market(config=env_config)

# ray.init(local_mode=True)
out = agent.evaluate()
print(out)


rewards = out['evaluation']['hist_stats']['episode_reward'] 
print(f'mean: {np.mean(rewards)}')
print(f'std: {np.std(rewards)}')

import matplotlib.pyplot as plt
# Convert rewards to DataFrame for easier plotting
rewards_df = pd.DataFrame(rewards, columns=['Rewards'])
# Create a boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=rewards_df)
plt.title('Boxplot of Rewards')
plt.savefig('boxplot_rewards.pdf')


# Create a histogram of rewards
plt.figure(figsize=(10, 6))
bins = np.arange(-5, 5, 0.25)
sns.histplot(data=rewards_df, x='Rewards', bins=bins, kde=True)
# sns.histplot(data=rewards_df, x='Rewards', bins=30, kde=True)
plt.title('Histogram of Rewards')
plt.savefig('histogram_rewards.pdf')


n = int(1e3)
# sample from market environment 
results = {}
rewards = []
M = Market(config=eval_config)
for n in range(n):
    if n%100 == 0:
        print(f'episode {n}')
    # print(f'episode {n}')
    observation, _ = M.reset()
    terminated = False 
    while not terminated:
        # action = M.action_space.sample()
        action = agent.compute_single_action(observation, explore=False)
        # print(action)        
        observation, reward, terminated, truncated, info = M.step(action)
        # print(f'reward: {reward}')
    rewards.append(info['total_reward'])

results['rl_mean'] = np.mean(rewards)
results['rl_std'] = np.std(rewards)


plt.figure(figsize=(10, 6))
bins = np.arange(-5, 5, 0.25)
rewards_df = pd.DataFrame(results['total_reward'], columns=['Rewards'])
sns.histplot(data=rewards_df, x='Rewards', bins=bins, kde=True)
# sns.histplot(data=rewards_df, x='Rewards', bins=30, kde=True)
plt.title('Histogram of Rewards')
plt.savefig('histogram_rewards.pdf')
print(results)


# # sample from market environment 
# print('------')
# rewards = []
# env_config['execution_agent'] = 'sl_agent'
# M = Market(config=env_config)
# for n in range(n):
#     if n%100 == 0:
#         print(f'episode {n}')
#     # print(f'episode {n}')
#     observation, _ = M.reset()
#     terminated = False 
#     while not terminated:
#         # action = M.action_space.sample()
#         # action = agent.compute_single_action(observation)
#         # print(action)        
#         observation, reward, terminated, truncated, info = M.step(action=None)
#         # print(f'reward: {reward}')
#     rewards.append(info['total_reward'])
# results['sl_mean'] = np.mean(rewards)
# results['sl_std'] = np.std(rewards)

# print('#####')
# print(results)
# results = pd.DataFrame(results, index=[0])
# print(results)
# results.to_csv('latest_results.csv')


