import numpy as np 
import os 
import sys 
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)
from ray import tune
from simulation.market_gym import Market, mp_rollout
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from ray.rllib.algorithms.algorithm import Algorithm
# from simulation.test_vectorization import mp_rollout
import ray 

ray.init()
# print('finished')

lots = [40]
envs = ['noise', 'flow']
n_cpus = 50

for n_lots in lots:
    # for env_type in ['strategic', 'flow', 'noise']:
    for env_type in envs:
        print('##############')
        print(env_type)
        print(n_lots)

        # env_type = 'strategic'
        # n_lots = 10
        data = {}

        # restore best agent 
        path = f'{parent_dir}/ray_results/{n_lots}_{env_type}'
        analysis = tune.ExperimentAnalysis(path)
        # analysis = tune.ExperimentAnalysis(path, default_metric="episode_reward_mean", default_mode="max")
        best_trial = analysis.get_best_trial(metric="episode_reward_mean", mode="max", scope="last")
        path = analysis.get_best_checkpoint(trial=best_trial, mode="max", metric="episode_reward_mean")
        print(f'BEST CHECKPOINT: {path}')
        agent = Algorithm.from_checkpoint(path)
        env_config = agent.evaluation_config.env_config
        print(f"env config for evalutation config is {env_config}")
        print(f"exploration is set to {agent.evaluation_config.explore}")
        out = agent.evaluate()
        ray.shutdown()
        rewards = out['evaluation']['hist_stats']['episode_reward'] 
        print('RL RESULTS')
        print(f'mean: {np.mean(rewards)}')
        # print(out['evaluation']['hist_stats']['episode_reward_mean'] )
        print(f'std: {np.std(rewards)}')

        data['E[rl]'] = np.mean(rewards)
        data['Std[rl]'] = np.std(rewards)
        
        # seeding ??
        sl, _, _ = mp_rollout(n_samples=1000, n_cpus=n_cpus, execution_agent='sl_agent', market_type=env_config['market_env'], volume=env_config['volume'])
        # sl = [x-1e-5 if x >=1 else x for x in sl]
        print('SL RESULTS')
        print(f'mean sl: {np.mean(sl)}')
        print(f'std sl: {np.std(sl)}')


        l_sl, _, _ = mp_rollout(n_samples=1000, n_cpus=50, execution_agent='linear_sl_agent', market_type=env_type, volume=n_lots)
        print('linear submit and leave:')
        print(f'mean l_sl: {np.mean(l_sl)}')
        print(f'std l_sl: {np.std(l_sl)}')

        data['E[sl]'] = np.mean(sl)
        data['Std[sl]'] = np.std(sl)
        data['E[l_sl]'] = np.mean(l_sl)
        data['Std[l_sl]'] = np.std(l_sl)

        # Create a dictionary from the data

        data = pd.DataFrame(data, index=[n_lots]).round(2)
        print(data)
        data.to_csv(f'rewards/{env_type}_{n_lots}.csv')

        # data = pd.DataFrame({'lsl': l_sl, 'sl': sl, 'rl': rewards})
        # print(data)
        # path = f'rewards/{env_type}_{n_lots}.csv'
        # print(f'saving to {path}')
        # # ordering will be reversed by seaborn. so that we have rl, sl, lsl 
        # data.to_csv(path)

        # Shut down all Ray actors
        # ray.shutdown()


# plt.figure(figsize=(10, 6))


# Plotting

# # Plotting individual distributions

# data = pd.DataFrame({'RL': rewards, 'S&L': sl, 'L S&L': l_sl})
# data = pd.DataFrame({'lsl': l_sl, 'sl': sl, 'rl': rewards})
# data.to_csv(f'{env_type}_{n_lots}.csv')


if False:
    plt.figure(figsize=(10, 6))
    # Plotting individual distributions
    # sns.histplot(data, kde=True, stat='probability', legend=True, multiple='dodge', binwidth=1, binrange=(-int(np.min(l_sl)), 7), shrink=0.8)
    # strtategic setting 40 lots 
    if env_type == 'strategic':
        sns.histplot(data, kde=True, stat='percent', legend=True, binwidth=2, binrange=(-18, 16), multiple='dodge', shrink=0.8, common_norm=False)
        plt.xticks(np.arange(-20, 20, 2))
        plt.xlim(-18, 16)
    elif env_type == 'flow':
        # sns.histplot(data, kde=True, stat='percent', legend=True, binwidth=2, binrange=(-18, 16), multiple='dodge', shrink=0.8, common_norm=False)
        if n_lots == 10:
            sns.histplot(data, kde=True, stat='percent', legend=True, binwidth=1, binrange=(-8, 7), multiple='dodge', shrink=0.8, common_norm=False)
            plt.xticks(np.arange(-20, 20, 1))
            plt.xlim(-8, 7)
        elif n_lots == 40:
            sns.histplot(data, kde=True, stat='percent', legend=True, binwidth=1, binrange=(-12, 5), multiple='dodge', shrink=0.8, common_norm=False)
            plt.xticks(np.arange(-20, 20, 1))
            plt.xlim(-12, 5)
        # sns.histplot(data, kde=True, stat='percent', legend=True, binwidth=2, binrange=(-13, 16), multiple='dodge', shrink=0.8, common_norm=False)
    elif env_type == 'noise':
        if n_lots == 10:
            sns.histplot(data, kde=True, stat='percent', legend=True, multiple='dodge', binwidth=1, binrange=(-9,6), shrink=0.8, common_norm=False)
            plt.xticks(np.arange(-20, 20, 1))
            plt.xlim(-9, 6)
        elif n_lots == 40:
            sns.histplot(data, kde=True, stat='percent', legend=True, multiple='dodge', binwidth=1, binrange=(-12,7), shrink=0.8, common_norm=False)
            plt.xticks(np.arange(-20, 20, 1))
            plt.xlim(-12, 7)
        # plt.xticks(np.arange(-20, 20, 1))
        # plt.xlim(-12, 5)        

    plt.legend(['l_sl', 'sl', 'rl'])
    plt.grid(True)
    plt.savefig(f'{env_type}_{n_lots}.pdf')

    # sns.histplot(sl, kde=True, color='blue', label='S&L')
    # sns.histplot(l_sl, kde=True, color='green', label='L S&L')
    # sns.histplot(rewards, kde=True, color='red', label='RL')
    # plt.legend()
    # plt.grid(True)
    # plt.xticks(np.arange(-10, 11, 1))
    # plt.xlim(-10, 7)
    # plt.ylim(0, 200)
    # plt.savefig('rl_sl_plots.pdf')



    # Convert rewards to DataFrame for easier plotting
    # rewards_df = pd.DataFrame(rewards, columns=['Rewards'])
    # # Create a boxplot
    # plt.figure(figsize=(10, 6))
    # sns.boxplot(data=rewards_df)
    # plt.title('Boxplot of Rewards')
    # plt.savefig('boxplot_rewards_0.pdf')


    ## Create a histogram of rewards
    # plt.figure(figsize=(10, 6))
    # bins = np.arange(-10, 10, 0.25)
    # sns.histplot(data=rewards_df, x='Rewards', bins=bins, kde=True)
    # # sns.histplot(data=rewards_df, x='Rewards', bins=30, kde=True)
    # plt.title('Histogram of Rewards')
    # plt.savefig('histogram_rewards_0.pdf')


