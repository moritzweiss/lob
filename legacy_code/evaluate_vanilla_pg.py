from ray import tune
from multiprocessing import Pool
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
# from advanced import Market
# from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.pg import PG
from ray.rllib.algorithms.pg import PGConfig
from ray.rllib.algorithms.ppo import PPOConfig
from custom_model import CustomTorchModel
from custom_model import TorchDirichlet
# from ray.rllib.models.torch.torch_action_dist import TorchDirichlet
from ray.rllib.models import ModelCatalog
from ray.rllib.models import MODEL_DEFAULTS
# import copy 
import time 
import numpy as np 
import ray 
from advanced_multi_lot import Market 
from ray.rllib.algorithms.algorithm import Algorithm
import sys
import os 
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


# model catalog 
# ModelCatalog.register_custom_model("my_torch_model", CustomTorchModel)
# custom_config = {}
# custom_config["vf_share_layers"] = False
# ModelCatalog.register_custom_action_dist("my_dist", TorchDirichlet)
# ray.init(local_mode=True)

# get best checkpoint and load agent 
# analysis = tune.ExperimentAnalysis('/u/weim/lob/results/100_lots_Gaussian_softmax_imbalance', default_metric="episode_reward_mean", default_mode="max")
# best_trial = analysis.get_best_trial(metric="episode_reward_mean", mode="max", scope="last")
# # other options: scope="last", scope="all"
# path = analysis.get_best_checkpoint(trial=best_trial, mode="max", return_path=True, metric="episode_reward_mean")
# agent = Algorithm.from_checkpoint(path)
# path = analysis.get_last_checkpoint(trial=best_trial).path 


# get environment config 
# config = analysis.get_best_config()
# config = {'total_n_steps': int(1e3), 'log': True, 'seed':0, 'initial_level': 2, 'initial_volume': 100, 'imbalance_trader': False}
# config['env_config']['seed'] = 0
# print(config['model']['custom_action_dist'])

# 

# naming: 250_imbalace, 250_simple, (#lots_imbalance, #lots_simple)
# same for 100 lots 


# n_lots 10 or 250
# env_type: imbalance or simple
# agent: all_passive, linear_passive, rl

class SampleStrategy():
    def __init__(self, n_lots=250, env_type='imbalance', strategy='linear_passive', n_samples=100, drift=None) -> None:
        assert strategy in ['all_passive', 'linear_passive', 'rl', 'submit_and_leave', 'submit_and_leave_linear']
        assert n_lots in [10, 100, 250]
        assert env_type in ['imbalance', 'simple', 'down']
        self.n_lots = n_lots
        self.env_type = env_type
        self.strategy = strategy
        self.n_samples = n_samples
        self.env_config = {'total_n_steps': int(1e3), 'log': False, 'seed': None, 'initial_level': 2, 'initial_volume': n_lots, 'env_type': env_type}
        if strategy == 'rl':
            path = f'{current}/results/{self.n_lots}_{self.env_type}'
            analysis = tune.ExperimentAnalysis(path, default_metric="episode_reward_mean", default_mode="max")
            best_trial = analysis.get_best_trial(metric="episode_reward_mean", mode="max", scope="last")
            config = analysis.get_best_config()
            path = analysis.get_best_checkpoint(trial=best_trial, mode="max", return_path=True, metric="episode_reward_mean")
            self.config = config 
            self.path = path 
    
    def sample_from_environment(self, seed=0):        
        # config = self.config
        # path = self.path
        # load config and path 
        # path = f'{current}/results/{self.n_lots}_{self.env_type}'
        # analysis = tune.ExperimentAnalysis(path, default_metric="episode_reward_mean", default_mode="max")
        # best_trial = analysis.get_best_trial(metric="episode_reward_mean", mode="max", scope="last")
        # config = analysis.get_best_config()
        # path = analysis.get_best_checkpoint(trial=best_trial, mode="max", return_path=True, metric="episode_reward_mean")
        # agent = Algorithm.from_checkpoint(path)
        # not sure if this actually works :) 
        # load agent 
        if self.strategy == 'rl':
            self.config['num_workers'] =  0 
            self.config['num_gpus'] =  0 
            AC = PPOConfig()
            AC = AC.update_from_dict(self.config)
            agent = AC.build()
            agent.restore(self.path)    
            # if self.strategy == 'linear_passive':
            #     v_delta = 25
            #     pass 
            # other options: scope="last", scope="all"
        self.env_config['seed'] = seed
        M = Market(config=self.env_config)
        rewards = []
        # print('start sampling')
        for n in range(self.n_samples):
            # draw ramdom either 'down' or 'up' from bernoulli distribution             
            # if self.drift == 'down':
            #     direction = 'down'
            # elif self.drift == 'random':
            #     direction = np.random.choice(['up', 'down'])
            # else:
            #     direction = None            
            reward_per_episode = 0 
            terminated = truncated = False
            if self.strategy == 'linear_passive' or self.strategy == 'submit_and_leave_linear':
                # this causes 25 orders to be placed in the book on rest 
                volume_delta = int(self.env_config['initial_volume']/10)
                M.initial_volume = volume_delta
                M.withhold_volume = self.env_config['initial_volume'] - volume_delta

            # observe the current order distribution
            # this send initial volume into the book 
            if self.strategy == 'submit_and_leave_linear':
                # dont care about this at the moment 
                observation, _ = M.reset(initialize_orders=False)
            else:
                observation, _ = M.reset(initialize_orders=True)
            while not terminated and not truncated:
                # actions 
                if self.strategy == 'rl':
                    action = agent.compute_single_action(observation, explore=False)
                elif self.strategy == 'all_passive':
                    action = np.array([-10, 10, -10, -10], dtype=np.float32)
                elif self.strategy == 'submit_and_leave':
                    action = np.array([-10, 10, -10, -10], dtype=np.float32)
                elif self.strategy == 'linear_passive':
                    # add 25 lots to the book 
                    action = np.array([-10, 10, -10, -10], dtype=np.float32)
                elif self.strategy == 'submit_and_leave_linear':
                    action = np.array([-10, 10, -10, -10], dtype=np.float32)
                else:
                    raise NotImplementedError
                # transitions 
                if self.strategy == 'submit_and_leave':
                    # ignore 
                    if M.time == 0:
                        observation, reward, terminated, truncated, info = M.step(action)
                    else:
                        observation, reward, terminated, truncated, info = M.step(action, no_action=True)
                elif self.strategy == 'submit_and_leave_linear':
                    observation, reward, terminated, truncated, info = M.step(action, additional_lots=volume_delta)
                else:
                    observation, reward, terminated, truncated, info = M.step(action)
                if (self.strategy == 'linear_passive' or self.strategy == 'submit_and_leave_linear')  and not truncated :
                    # add more volume 
                    M.volume += volume_delta
                    M.withhold_volume -= volume_delta
                reward_per_episode += reward
                if self.strategy == 'linear_passive' or self.strategy == 'submit_and_leave_linear':
                    pass
                else:
                    assert observation in M.observation_space
                # assert observation in M.observation_space
            if self.strategy == 'linear_passive' or self.strategy == 'submit_and_leave_linear':
                # initial volume is 25
                rewards.append(reward_per_episode*0.1)
            else:
                rewards.append(reward_per_episode)
        return rewards
    
    def multi_process_sample(self, n_workers=10, seed=0):
        seeds = [s+seed for s in range(n_workers)]
        t = time.time()
        p = Pool(n_workers)
        out = p.map(self.sample_from_environment, seeds)
        print(f'time for simulation: {time.time()-t}s')
        # print(out)
        print('DONE')
        rewards = np.concatenate(out)   

        print(f'total number of episodes: {len(rewards)}')
        print(f'the mean reward is {np.mean(rewards)}')
        print(f'the min reward is {np.min(rewards)}')
        print(f'the max reward is {np.max(rewards)}')
        
        np.save(f'data/rewards_{self.n_lots}_{self.strategy}_{self.env_type}', rewards)
        # e.g. rewards_100_rl_imbalance.npy
        return rewards

# S = SampleStrategy(n_lots=250, env_type='imbalance', strategy='all_passive', n_samples=250, drift='down')
# S.sample_from_environment(seed=0)

# agent: all_passive, linear_passive, rl
# 'all_passive', 'linear_passive']
for n_lots in [10, 250]:
    # for strategy in ['rl', 'all_passive', 'linear_passive']:
    for strategy in ['rl', 'linear_passive', 'all_passive']:
        for env_type in ['simple', 'imbalance', 'down']:
            # print(S.drift)
            print(f'currently simulating {n_lots}_{strategy}_{env_type}')
            S = SampleStrategy(n_lots=n_lots, env_type=env_type, strategy=strategy, n_samples=250)
            S.multi_process_sample(n_workers=20, seed=0)


# n_workers = 10
# seeds = range(n_workers)
# seeds = [s+10 for s in seeds]
# t = time.time()
# p = Pool(n_workers)
# out = p.map(S.sample_from_environment, seeds)
# print(f'time for simulation: {time.time()-t}s')
# # print(out)
# print('DONE')
# rewards = np.concatenate(out)   

# print(f'total number of episodes: {len(rewards)}')
# print(f'the mean reward is {np.mean(rewards)}')
# print(f'the min reward is {np.min(rewards)}')
# print(f'the max reward is {np.max(rewards)}')

# result with explore and 4000 steps is 6.568
# reult without exploration (softmax for deterministic action) and 4000 steps is 6.91725
# benchmark policy and 4000 steps is 6.91725 
# result without exploration and built in mean function is 6.574 
# adding softmax improves the result dramaticall

# optimal result should be around 6.9 
# elapsed = time.time()-start
# print(f'time elapsed in seconds: {elapsed}')
# print(max(rewards))
# print(min(rewards))
# print(len(rewards))
# print(np.mean(rewards))

# rewards = np.array(rewards)
# np.save('rewards_100_lots_Gaussian', rewards)


# x= np.load('rewards_algo_imbalance.npy')
# print(x)

# evaluation results: 
# the result was -0.21
# compared to -0.252 (benchmark. submit and leave)

# for seed = 0
# for 1000 iterations we obtain 
# this was the algo with queue position and shortfall knowledge 
# algo = - 0.147
# benchmark =  -0.273


### and the algo with queue position 
# for seed = 0
# for 1000 iterations the result was -0.177
# for 2000 iterations the resilt was -0.1435, on a different run it was -0.1275, 
# yet a different run -0.1365

# benchmark is -0.269

# also try without exploration 