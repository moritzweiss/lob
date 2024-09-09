# legacy code 

from agents import NoiseAgent, LinearSubmitLeaveAgent, StrategicAgent, SubmitAndLeaveAgent, MarketAgent, InitialAgent
from limit_order_book.limit_order_book import LimitOrderBook
from config.config import noise_agent_config, strategic_agent_config, sl_agent_config, linear_sl_agent_config, market_agent_config, initial_agent_config
import numpy as np
import pandas as pd 
from config.config import noise_agent_config
from queue import PriorityQueue
from dataclasses import dataclass, field
from typing import Any
from multiprocessing import Pool
import itertools
import time


class Market():
    def __init__(self, market_env='noise', execution_agent='sl_agent', volume=10, seed=0):
        
        assert market_env in ['noise', 'flow', 'strategic']
        assert execution_agent in ['market_agent', 'sl_agent', 'linear_sl_agent']

        self.agents = {}
        
        # initial agent         
        if market_env == 'noise':
            initial_agent_config['initial_shape_file'] = 'initial_shape/noise.npz'
        else:
            # initial_agent_config['initial_shape_file'] = 'initial_shape/noise_unit.npz'
            initial_agent_config['initial_shape_file'] = 'initial_shape/noise_flow_75.npz'
        agent = InitialAgent(**initial_agent_config)
        self.agents[agent.agent_id] = agent

        # noise agent 
        noise_agent_config['rng'] = np.random.default_rng(seed)
        noise_agent_config['unit_volume'] = False
        noise_agent_config['terminal_time'] = 150
        noise_agent_config['start_time'] = 0 
        noise_agent_config['fall_back_volume'] = 5
        # TODO: make start time more consistent 
        if market_env == 'noise':
            noise_agent_config['imbalance_reaction'] = False
            agent = NoiseAgent(**noise_agent_config)
            self.agents[agent.agent_id] = agent
        else: 
            noise_agent_config['imbalance_reaction'] = True
            noise_agent_config['imbalance_factor'] = 2.0
            agent = NoiseAgent(**noise_agent_config)            
        # TODO: make those intensity adjustments automatically 
            agent.limit_intensities = agent.limit_intensities * 0.85
            agent.market_intensity = agent.market_intensity * 0.85
            agent.cancel_intensities = agent.cancel_intensities * 0.85
            self.agents[agent.agent_id] = agent

        # strategic agent 
        if market_env == 'strategic':
            strategic_agent_config['time_delta'] = 7.5
            strategic_agent_config['market_volume'] = 1
            strategic_agent_config['limit_volume'] = 1
            strategic_agent_config['rng'] = np.random.default_rng(seed)
            agent = StrategicAgent(**strategic_agent_config)
            self.agents[agent.agent_id] = agent 

        # execution agent
        if execution_agent == 'market_agent':
            sl_agent_config['start_time'] = 0
            market_agent_config['volume'] = volume
            agent = MarketAgent(**market_agent_config)
        elif execution_agent == 'sl_agent':
            sl_agent_config['start_time'] = 0
            sl_agent_config['volume'] = volume
            sl_agent_config['terminal_time'] = 150
            agent = SubmitAndLeaveAgent(**sl_agent_config)
        else: 
            linear_sl_agent_config['start_time'] = 0
            linear_sl_agent_config['volume'] = volume
            linear_sl_agent_config['terminal_time'] = 150
            linear_sl_agent_config['time_delta'] = 15
            agent = LinearSubmitLeaveAgent(**linear_sl_agent_config)
        self.agents[agent.agent_id] = agent
        self.execution_agent_id = agent.agent_id
        return None 


    def reset(self):
        list_of_agents = list(self.agents.keys())
        self.lob = LimitOrderBook(list_of_agents=list_of_agents, level=30, only_volumes=False)
        # reset agents 
        for agent_id in self.agents:
            self.agents[agent_id].reset()
        # initialize event queue 
        self.pq = PriorityQueue()
        # set initial events 
        for agent_id in self.agents:
            # noise agent just puts the first event at its initial time
            # although this is not completely correct 
            out = self.agents[agent_id].initial_event()
            self.pq.put(out)
        return None 
    
    def run(self):
        n_events = 0  
        # initial_bid = self.lob.get_best_price('bid')
        terminated = False
        # observation = False
        while not self.pq.empty() and not terminated: 
            n_events += 1
            time, _, event = self.pq.get()
            orders = self.agents[event].generate_order(lob=self.lob, time=time)
            msgs = self.lob.process_order_list(orders)
            # check if the simulation is terminated because the execution agent got filled 
            _, terminated = self.agents[self.execution_agent_id].update_position_from_message_list(msgs)
            if terminated:
                break
            # if not terminated or execution agent not present, generate a new event 
            # can be None if there are no more events happening for the agent 
            out = self.agents[event].new_event(time, event)
            if out is not None:
                self.pq.put(out)

        return self.agents[self.execution_agent_id].cummulative_reward, self.agents[self.execution_agent_id].limit_sells/self.agents[self.execution_agent_id].initial_volume, n_events  



def rollout(seed, num_episodes, execution_agent, market_type, volume):
    M = Market(volume=volume, execution_agent=execution_agent, market_env=market_type, seed=seed)
    total_rewards = []
    fill_rates = []
    n_events = []
    for _ in range(num_episodes):
        M.reset()
        total_reward, fill_rate, n_event = M.run()
        total_rewards.append(total_reward)
        fill_rates.append(fill_rate)
        n_events.append(n_event)
    return total_rewards, fill_rates, n_events

# note: can also use ray for multiprocessing rollouts 
def mp_rollout(n_samples, n_cpus, execution_agent, market_type, volume):
    samples_per_env = int(n_samples/n_cpus) 
    with Pool(n_cpus) as p:
        out = p.starmap(rollout, [(seed, samples_per_env, execution_agent, market_type, volume) for seed in range(n_cpus)])    
    all_rewards, passive_fills, n_events  = zip(*out)
    all_rewards = list(itertools.chain.from_iterable(all_rewards))
    passive_fills = list(itertools.chain.from_iterable(passive_fills))
    n_events = list(itertools.chain.from_iterable(n_events))
    # n_events = list(itertools.chain.from_iterable(n_events))
    # n_cancel = list(itertools.chain.from_iterable(n_cancel))
    # n_limit = list(itertools.chain.from_iterable(n_limit))
    # n_market = list(itertools.chain.from_iterable(n_market))
    # drift = list(itertools.chain.from_iterable(drift))
    return all_rewards, passive_fills, n_events


if __name__ == '__main__':

    # start_time = time.time()
    # rewards, fill_rates, n_events = rollout(seed=0, num_episodes=10, execution_agent='sl_agent', market_type='flow', volume=10)
    # end_tine = time.time()
    # execution_time = end_tine - start_time
    # print("Execution time:", execution_time)
    # print(rewards)
    # print(fill_rates)

    # n_samples = 100
    # n_cpus = 10
    # agent = 'linear_sl_agent'
    # env = 'strategic'
    # lots = 40
    # start_time = time.time()
    # rewards, fill_rates, n_events = mp_rollout(n_samples, n_cpus, agent, env, lots)
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print("Execution time:", execution_time)
    # print(rewards)
    # print(fill_rates)



    envs = ['noise', 'flow', 'strategic']
    n_samples = 1000
    n_cpus = 70
    start_time = time.time()
    for lots in [10, 40]:
        # print(lots)
        results = {}
        for agent in ['sl_agent', 'linear_sl_agent']:
            results[f'{agent}_reward_mean'] = []
            results[f'{agent}_reward_std'] = []
            results[f'{agent}_pfill_rate'] = []
            results[f'{agent}_n_events'] = []
            for env in envs:
                rewards, fill_rate, n_events = mp_rollout(n_samples, n_cpus, agent, env, lots)
                results[f'{agent}_reward_mean'].append(np.mean(rewards))
                results[f'{agent}_reward_std'].append(np.std(rewards))
                results[f'{agent}_pfill_rate'].append(np.mean(fill_rate))
                results[f'{agent}_n_events'].append(np.mean(n_events))
        end_time = time.time()
        results = pd.DataFrame.from_dict(results).round(2)
        results.index = envs
        print(results)
        results.to_csv(f'results/benchmarks_{lots}.csv')
    execution_time = end_time - start_time
    print("Execution time:", execution_time)

