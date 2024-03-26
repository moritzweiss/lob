import numpy as np 
import os 
import sys 
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)
from ray import tune
from simulation.all_markets_simulation import Market
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from ray.rllib.algorithms.algorithm import Algorithm


# restore best agent 
path = f'{parent_dir}/ray_results/40_noise'
analysis = tune.ExperimentAnalysis(path, default_metric="episode_reward_mean", default_mode="max")
best_trial = analysis.get_best_trial(metric="episode_reward_mean", mode="max", scope="last")
path = analysis.get_best_checkpoint(trial=best_trial, mode="max", metric="episode_reward_mean")
print(f'BEST CHECKPOINT: {path}')
agent = Algorithm.from_checkpoint(path)
env_config = agent.evaluation_config.env_config
print(f"env config for evalutation config is {env_config}")
print(f"exploration is set to {agent.evaluation_config.explore}")

out = agent.evaluate()
rewards = out['evaluation']['hist_stats']['episode_reward'] 
print(f'mean: {np.mean(rewards)}')
print(f'std: {np.std(rewards)}')

# Convert rewards to DataFrame for easier plotting
rewards_df = pd.DataFrame(rewards, columns=['Rewards'])
# Create a boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=rewards_df)
plt.title('Boxplot of Rewards')
plt.savefig('boxplot_rewards_0.pdf')


## Create a histogram of rewards
plt.figure(figsize=(10, 6))
bins = np.arange(-10, 10, 0.25)
sns.histplot(data=rewards_df, x='Rewards', bins=bins, kde=True)
# sns.histplot(data=rewards_df, x='Rewards', bins=30, kde=True)
plt.title('Histogram of Rewards')
plt.savefig('histogram_rewards_0.pdf')



""" 
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
print(results)
 """
""" 

# print('#####')
# print(results)
# print(results)


plt.figure(figsize=(10, 6))
bins = np.arange(-5, 5, 0.25)
# rewards = pd.DataFrame(rewards, index=[0])
rewards = pd.DataFrame(rewards, columns=['Rewards'])
sns.histplot(data=rewards, x='Rewards', bins=bins, kde=True)
# sns.histplot(data=rewards_df, x='Rewards', bins=30, kde=True)
plt.title('Histogram of Rewards')
plt.savefig('histogram_rewards.pdf')
rewards.to_csv('latest_results.csv') """


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


