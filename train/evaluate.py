import numpy as np 
import os 
from os import environ
import sys 
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)
from ray import tune
# N_THREADS = '1'
# environ['OMP_NUM_THREADS'] = N_THREADS
# environ['OPENBLAS_NUM_THREADS'] = N_THREADS
# environ['MKL_NUM_THREADS'] = N_THREADS
# environ['VECLIB_MAXIMUM_THREADS'] = N_THREADS
# environ['NUMEXPR_NUM_THREADS'] = N_THREADS
from simulation.all_markets_simulation import Market, config
from ray.rllib.algorithms.ppo import PPOConfig
import pandas as pd 

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


n = int(1e3)
# sample from market environment 
results = {}
rewards = []
M = Market(config=env_config)
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


# sample from market environment 
print('------')
rewards = []
env_config['execution_agent'] = 'sl_agent'
M = Market(config=env_config)
for n in range(n):
    if n%100 == 0:
        print(f'episode {n}')
    # print(f'episode {n}')
    observation, _ = M.reset()
    terminated = False 
    while not terminated:
        # action = M.action_space.sample()
        # action = agent.compute_single_action(observation)
        # print(action)        
        observation, reward, terminated, truncated, info = M.step(action=None)
        # print(f'reward: {reward}')
    rewards.append(info['total_reward'])
results['sl_mean'] = np.mean(rewards)
results['sl_std'] = np.std(rewards)

print('#####')
print(results)
results = pd.DataFrame(results, index=[0])
print(results)
results.to_csv('latest_results.csv')


