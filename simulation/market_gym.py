import os, sys 
current_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_path)
sys.path.append(parent_dir)
from simulation.agents import NoiseAgent, LinearSubmitLeaveAgent, StrategicAgent, SubmitAndLeaveAgent, MarketAgent, InitialAgent, ObservationAgent, RLAgent
from limit_order_book.limit_order_book import LimitOrderBook
from config.config import noise_agent_config, strategic_agent_config, sl_agent_config, linear_sl_agent_config, market_agent_config, initial_agent_config, observation_agent_config, rl_agent_config
import numpy as np
import pandas as pd 
from config.config import noise_agent_config
from queue import PriorityQueue
from dataclasses import dataclass, field
from typing import Any
from multiprocessing import Pool
import itertools
import time
import gymnasium as gym 
import torch

class Market(gym.Env):
    # TODO: write two seperate classes: one for benchmark agents and one for rl agents 
    # the rollouts are anyways separate. this could make things more neat 
    # write up what happens for both classes on paper 
    def __init__(self, config):

        """
        - config should have keys market_env, execution_agent, volume, seed
        - we use a config, becuase this is required by rl lib 
        - seed will be set depending on whether we use multiple workers or not 
        """

        assert 'market_env' in config
        assert 'execution_agent' in config
        assert 'volume' in config
        assert 'seed' in config

        seed = config['seed']
        
        assert config['market_env'] in ['noise', 'flow', 'strategic']
        assert config['execution_agent'] in ['market_agent', 'sl_agent', 'linear_sl_agent', 'rl_agent']

        self.agents = {}
        
        # initial agent         
        if config['market_env'] == 'noise':
            initial_agent_config['initial_shape_file'] = 'initial_shape/noise_unit.npz'
        else:
            initial_agent_config['initial_shape_file'] = 'initial_shape/noise_unit.npz'
            # initial_agent_config['initial_shape_file'] = 'initial_shape/noise_flow_75.npz'
        agent = InitialAgent(**initial_agent_config)
        self.agents[agent.agent_id] = agent

        # noise agent 
        noise_agent_config['rng'] = np.random.default_rng(seed)
        noise_agent_config['unit_volume'] = False
        noise_agent_config['terminal_time'] = 150
        noise_agent_config['start_time'] = 0 
        noise_agent_config['fall_back_volume'] = 5
        # TODO: make start time more consistent 
        if config['market_env'] == 'noise':
            noise_agent_config['imbalance_reaction'] = False
            agent = NoiseAgent(**noise_agent_config)
            self.agents[agent.agent_id] = agent
        else: 
            noise_agent_config['imbalance_reaction'] = True
            noise_agent_config['imbalance_factor'] = 2.0
            agent = NoiseAgent(**noise_agent_config)            
            # TODO: make those intensity adjustments automatically or move them to the config file 
            agent.limit_intensities = agent.limit_intensities * 0.85
            agent.market_intensity = agent.market_intensity * 0.85
            agent.cancel_intensities = agent.cancel_intensities * 0.85
            self.agents[agent.agent_id] = agent

        # strategic agent 
        if config['market_env'] == 'strategic':
            strategic_agent_config['time_delta'] = 7.5
            strategic_agent_config['market_volume'] = 1
            strategic_agent_config['limit_volume'] = 1
            strategic_agent_config['rng'] = np.random.default_rng(seed)
            agent = StrategicAgent(**strategic_agent_config)
            self.agents[agent.agent_id] = agent 

        # execution agent
        if config['execution_agent'] == 'market_agent':
            sl_agent_config['start_time'] = 0
            market_agent_config['volume'] = config['volume']
            agent = MarketAgent(**market_agent_config)
        elif config['execution_agent'] == 'sl_agent':
            sl_agent_config['start_time'] = 0
            sl_agent_config['volume'] = config['volume']
            sl_agent_config['terminal_time'] = 150
            agent = SubmitAndLeaveAgent(**sl_agent_config)
        elif config['execution_agent'] == 'linear_sl_agent': 
            linear_sl_agent_config['start_time'] = 0
            linear_sl_agent_config['volume'] = config['volume']
            linear_sl_agent_config['terminal_time'] = 150
            linear_sl_agent_config['time_delta'] = 15
            agent = LinearSubmitLeaveAgent(**linear_sl_agent_config)
        else:
            rl_agent_config['start_time'] = 0
            rl_agent_config['terminal_time'] = 150
            rl_agent_config['time_delta'] = 15
            rl_agent_config['volume'] = config['volume']
            agent = RLAgent(**rl_agent_config)

        self.agents[agent.agent_id] = agent
        self.execution_agent_id = agent.agent_id

        # observation agent if rl agent is present: this will interupt the kernel at some time interval 
        if config['execution_agent'] == 'rl_agent':
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(agent.observation_space_length,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-10, high=10, shape=(agent.action_space_length,), dtype=np.float32)    
            agent = ObservationAgent(**observation_agent_config)
            self.agents[agent.agent_id] = agent
        else:
            # this is just dummy observation space, to make vectorized environments work for benchmark agents 
            # this is a bit clumsy 
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        return None 


    def reset(self, seed=None, options=None):
        self.lob = LimitOrderBook(list_of_agents=list(self.agents.keys()), level=30, only_volumes=False)
        # reset agents 
        for agent_id in self.agents:
            self.agents[agent_id].reset()
        # initialize event queue 
        self.pq = PriorityQueue()
        # set initial events 
        for agent_id in self.agents:
            # noise agent just puts the first event at its initial time (the first event should also be drawn from an exponential distribution)
            # although this is not completely correct 
            out = self.agents[agent_id].initial_event()
            self.pq.put(out)
        # runs up to first obsercation or stops if the simulation is terminated 
        # if there is no RL agent present: transition will just run straight to the end without any intermediate action
        observation, reward, terminated, info = self.transition()
        # this will terminate if the execution agent is one of the benchmark agents 
        return observation, info
    
    def step(self, action=None):
        observation, reward ,terminated, info = self.transition(action)
        if terminated:
            assert self.agents[self.execution_agent_id].volume == 0
        return observation, reward, terminated, False, info 

    def transition(self, action=None):
        terminated = False
        transition_reward = 0 
        # n_events = 0  
        while not self.pq.empty(): 
            # get next event from the event queue 
            time, _, agent_id = self.pq.get()
            if time > self.agents[self.execution_agent_id].terminal_time:
                # simulation should terminate at the execution agents terminal time
                raise ValueError("time is greater than execution agents terminal time")
            if agent_id == 'rl_agent':
                # if rl agent is present, generate an order based on action 
                orders = self.agents[agent_id].generate_order(lob=self.lob, time=time, action=action)
            else:
                # for the benchmark agents, no action is needed 
                orders = self.agents[agent_id].generate_order(lob=self.lob, time=time)
            # update order book, and check whether execution agent orders have been filled 
            # orders can be None if the agent has no more events in the pipeline. in this case generate_order() returns None 
            if orders is not None:
                msgs = self.lob.process_order_list(orders)
                reward, terminated = self.agents[self.execution_agent_id].update_position_from_message_list(msgs)
                transition_reward += reward
                if terminated:
                    break
            # if not terminated or execution agent not present, generate a new event 
            # can be None if there are no more events happening for the agent 
            out = self.agents[agent_id].new_event(time, agent_id)
            if out is not None:
                self.pq.put(out)
            # observation agent breaks the loop 
            if agent_id == 'observation_agent':
                break
        # if terminated:        
        # if self.pq.empty() or terminated:
        if terminated:
            assert self.agents[self.execution_agent_id].volume == 0
        info = {'cum_reward': self.agents[self.execution_agent_id].cummulative_reward, 
                'passive_fill_rate': self.agents[self.execution_agent_id].limit_sells/self.agents[self.execution_agent_id].initial_volume,                
                'time': time,
                # 'drift': (self.lob.data.best_bid_prices[-1] + self.lob.data.best_ask_prices[-1])/2 - self.reference_mid_price,
                'n_events': self.agents['noise_agent'].n_events,
                'terminated': terminated
                # 'terminated': terminated
                }
        if self.execution_agent_id == 'rl_agent':
            observation = self.agents[self.execution_agent_id].get_observation(time, self.lob)
        else:
            # benchmark agents should return None as observation 
            # assert observation is None
            observation = np.array([None], dtype=np.float32)
        return observation, transition_reward, terminated, info 


def make_env(config):
    def thunk():
        return Market(config)
    return thunk

def rollout_vectorized_rl(n_episodes, env_fns,
                          Model, model_path, device 
                          ):
    """        
        - loads rl policy and rolls it out in a vectorized manner
        - seed is used via seed+s (s in range(n_envs))

    note:
        - choosing cpu or cuda as device doesn't seem to make any difference 
        - we can max out the n_envs. for some reason one can even choose n_envs > n_proc without any errors                 
    """
    # load environment and model
    env = gym.vector.AsyncVectorEnv(env_fns)
    agent = Model(env).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()
    # prepare rollout
    rewards = []
    times = []
    n_events = []
    # start sampling 
    observation, info = env.reset()
    while len(rewards) < n_episodes:
        action, _, _, _ = agent.get_action_and_value(torch.Tensor(observation).to(device))
        observation, reward, terminated, truncated, infos = env.step(action.cpu().numpy())
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    rewards.append(info['cum_reward'])
    return rewards


def rollout_vectorized_benchmarks(seed, n_episodes, n_envs, execution_agent, market_type, volume): 
    """
        - uses vectorized environment to roll out benchmark rewards 
        - seed is used via seed+s (s in range(n_envs))

    note: 
        - mp_rollout is faster for benchmark agents but slower for the rl agent 
            (maybe because of the forward pass in the neural network)        
    """
    assert execution_agent != 'rl_agent'    
    samples_per_env = int(n_episodes/n_envs) 
    configs = [{'seed': seed+s, 'market_env': market_type, 'execution_agent': execution_agent, 'volume': volume} for s in range(n_envs)]
    env_fns = [make_env(c) for c in configs]
    M = gym.vector.AsyncVectorEnv(env_fns) 
    total_rewards = []
    times = []
    n_events = []
    for _ in range(samples_per_env):
        _, info = M.reset()
        total_rewards.extend(list(info['cum_reward']))
        times.append(list(info['time']))
        n_events.append(list(info['n_events']))
    return total_rewards, times, n_events



def rollout(seed, n_episodes, execution_agent, market_type, volume):
    """
        - rolls out the environment for n_episodes
        - returns the total rewards, times, and number of events
        - if execution_agent is rl_agent we generate some action and step through the environment 
        - otherwise, we just reset and the environment runs to the end without any intermediate action 
    """
    config = {'seed': seed, 'market_env': market_type, 'execution_agent': execution_agent, 'volume': volume}
    M = Market(config)
    total_rewards = []
    times = []
    n_events = []
    for _ in range(n_episodes):
        # this runs through the end for the benchmark agents 
        observation, info = M.reset()
        if execution_agent == 'rl_agent':
            terminated = False
            while not terminated:
                action = np.array([-10, -10, 10, -10, -10], dtype=np.float32)
                assert action in M.action_space
                observation, reward, terminated, truncated, info = M.step(action)
                assert observation in M.observation_space
        # print(info)
        total_rewards.append(info['cum_reward'])
        times.append(info['time'])
        n_events.append(info['n_events'])
    return total_rewards, times, n_events


def mp_rollout(n_samples, n_cpus, execution_agent, market_type, volume):
    """
        - is slightly faster than a vectorized rollout 
        - this runs rollout for each cpu     
        - each cpu runs n_samples/n_cpus episodes with a different seed 
        - returns rewards, times, events 
    """
    samples_per_env = int(n_samples/n_cpus) 
    with Pool(n_cpus) as p:
        # seed+1, in order to match worker_index
        out = p.starmap(rollout, [(200+seed, samples_per_env, execution_agent, market_type, volume) for seed in range(n_cpus)])    
    all_rewards, times, n_events  = zip(*out)
    all_rewards = list(itertools.chain.from_iterable(all_rewards))
    times = list(itertools.chain.from_iterable(times))
    n_events = list(itertools.chain.from_iterable(n_events))
    return all_rewards, times, n_events


if __name__ == '__main__':

    n_samples = 1000
    n_cpus = 80
    agent = 'rl_agent'
    env = 'flow'
    lots = 40
    seed = 100

    # start_time = time.time()
    # rewards, times, n_events = rollout(seed=0, n_episodes=10, execution_agent='linear_sl_agent', market_type='flow', volume=40)
    # end_tine = time.time()
    # execution_time = end_tine - start_time
    # print("Execution time:", execution_time)
    # print(f'rewards: {rewards}')
    # print(f'times: {times}')

    # start_time = time.time()
    # rewards, times, n_events = mp_rollout(n_samples, n_cpus, agent, env, lots)
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print("Execution time:", execution_time)
    # # print(rewards)
    # print(f'mean rewards: {np.mean(rewards)}')
    # print(fill_rates)


    # # rollout rl policy 
    # configs = [{'seed': seed+s, 'market_env': env, 'execution_agent': 'rl_agent', 'volume': lots} for s in range(n_cpus)]
    # env_fns = [make_env(c) for c in configs]
    # from rl_files.ppo_continuous_action import Agent
    # model_path = "runs/Market__ppo_continuous_action__1__1722434240_large_batch_flow40/ppo_continuous_action.cleanrl_model"
    # device = torch.device("cpu")
    # start_time = time.time()
    # total_rewards = rollout_vectorized_rl(n_episodes=n_samples, env_fns=env_fns, Model=Agent, model_path=model_path, device=device)
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print("Execution time:", execution_time)
    # print(f'mean rewards: {np.mean(total_rewards)}')


    # start_time = time.time()
    # # this is only about 1 second slower than mp rollout for 100 samples and 80 cpus
    # total_rewards, times, n_events = rollout_vectorized_benchmarks(seed=0, n_episodes=n_samples, execution_agent='linear_sl_agent',  n_envs=n_cpus, market_type=env, volume=lots)
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print("Execution time:", execution_time)
    # # print(total_rewards)
    # print(f'mean rewards: {np.mean(total_rewards)}')


    # envs = ['noise', 'flow', 'strategic']
    # n_samples = 10
    # n_cpus = 2
    # start_time = time.time()
    # for lots in [10, 40]:
    #     # print(lots)
    #     results = {}
    #     for agent in ['sl_agent', 'linear_sl_agent']:
    #         results[f'{agent}_reward_mean'] = []
    #         results[f'{agent}_reward_std'] = []
    #         # results[f'{agent}_pfill_rate'] = []
    #         results[f'{agent}_n_events'] = []
    #         for env in envs:
    #             rewards, times, n_events = mp_rollout(n_samples, n_cpus, agent, env, lots)
    #             results[f'{agent}_reward_mean'].append(np.mean(rewards))
    #             results[f'{agent}_reward_std'].append(np.std(rewards))
    #             results[f'{agent}_n_events'].append(np.mean(n_events))
    #             # results[f'{agent}_pfill_rate'].append(np.mean(fill_rate))
    #     end_time = time.time()
    #     results = pd.DataFrame.from_dict(results).round(2)
    #     results.index = envs
    #     print(results)
    #     results.to_csv(f'results/benchmarks_{lots}.csv')
    # execution_time = end_time - start_time
    # print("Execution time:", execution_time)