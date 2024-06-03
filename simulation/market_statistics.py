from agents import NoiseAgent, LinearSubmitLeaveAgent, StrategicAgent, SubmitAndLeaveAgent, MarketAgent, RLAgent, InitialAgent
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
import seaborn as sns
import matplotlib.pyplot as plt


class Market():
    def __init__(self, market_env='noise', seed=0):
        assert market_env in ['noise', 'flow', 'strategic']

        self.agents = {}

        # initial agent setting 
        initial_agent_config['initial_shape_file'] = 'initial_shape/noise_unit.npz'
        initial_agent_config['start_time'] = 0
        agent = InitialAgent(**initial_agent_config)
        self.agents[agent.agent_id] = agent

        # noise agent setting
        noise_agent_config['rng'] = np.random.default_rng(seed)
        noise_agent_config['unit_volume'] = False
        noise_agent_config['terminal_time'] = 150
        noise_agent_config['start_time'] = 0
        if market_env == 'noise':
            noise_agent_config['imbalance_reaction'] = False
            noise_agent_config['initial_shape_file'] = 'initial_shape/noise_unit.npz'
            agent = NoiseAgent(**noise_agent_config)
        else: 
            noise_agent_config['imbalance_reaction'] = True
            noise_agent_config['imbalance_factor'] = 2.0
            noise_agent_config['damping_factor'] = 0.75
            agent = NoiseAgent(**noise_agent_config)
            # noise_agent_config['initial_shape_file'] = 'initial_shape/noise_flow_75_unit.npz'
            agent.limit_intensities = agent.limit_intensities * 0.85
            agent.market_intensity = agent.market_intensity * 0.85
            agent.cancel_intensities = agent.cancel_intensities * 0.85
        self.agents[agent.agent_id] = agent
        
        # strategic agent setting 
        if market_env == 'strategic':
            strategic_agent_config['time_delta'] = 7.5
            strategic_agent_config['market_volume'] = 1
            strategic_agent_config['limit_volume'] = 1
            strategic_agent_config['rng'] = np.random.default_rng(seed)
            agent = StrategicAgent(**strategic_agent_config)
            self.agents[agent.agent_id] = agent 
        
    

    def reset(self):
        list_of_agents = list(self.agents.keys()) 
        self.lob = LimitOrderBook(list_of_agents=list_of_agents, level=30, only_volumes=False)
        for agent_id in list_of_agents:
            self.agents[agent_id].reset()            
        if 'strategic_agent' in self.agents:
            self.agents['strategic_agent'].direction = 'sell'
        self.pq = PriorityQueue()
        for agent_id in self.agents:
            out = self.agents[agent_id].initial_event()
            self.pq.put(out)
        return None
    
    def run(self):
        n_events = 0 
        n_cancellations = 0 
        n_limits = 0 
        n_markets = 0 
        event = None
        while not self.pq.empty(): 
            n_events += 1
            time, _, event = self.pq.get()
            orders = self.agents[event].generate_order(lob=self.lob, time=time)
            self.lob.process_order_list(orders)
            out = self.agents[event].new_event(time, event)
            if out is not None:
                self.pq.put(out)
            if event ==  'initial_agent':
                initial_mid = (self.lob.get_best_price('bid')+self.lob.get_best_price('ask'))/2
        
        drift = (self.lob.get_best_price('bid')+self.lob.get_best_price('ask'))/2 - initial_mid

        return time, n_events, drift  


def rollout(seed, num_episodes, market_type):
    M = Market(market_env=market_type, seed=seed)
    n_events = []
    drifts = []
    times = []
    for _ in range(num_episodes):
        M.reset()
        time, n_event, drift = M.run()
        n_events.append(n_event)
        drifts.append(drift)
        times.append(time)
    return n_events, drifts, times 


def mp_rollout(n_samples, n_cpus, market_type):
    samples_per_env = int(n_samples/n_cpus) 
    with Pool(n_cpus) as p:
        out = p.starmap(rollout, [(seed, samples_per_env, market_type) for seed in range(n_cpus)])    
    n_events, drifts, times  = zip(*out)
    n_events = list(itertools.chain.from_iterable(n_events))
    times = list(itertools.chain.from_iterable(times))
    drifts = list(itertools.chain.from_iterable(drifts))
    return n_events, drifts, times  

out = rollout(seed=0, num_episodes=10, market_type='strategic')
print(out)

out = mp_rollout(100, 10, 'flow')
print(out)


if __name__ == '__main__':
    envs = ['noise', 'flow', 'strategic']
    n_samples = 1000
    n_cpus = 50
    results = {}
    results[f'n_events'] = []
    results[f'drift_mean'] = []
    results[f'drift_std'] = []
    data_for_plot = {}
    for env in envs:
        n_events, drifts, times = mp_rollout(n_samples, n_cpus, env)
        data_for_plot[env] = drifts
        results[f'n_events'].append(np.mean(n_events))
        results[f'drift_mean'].append(np.mean(drifts))
        results[f'drift_std'].append(np.std(drifts))
    results = pd.DataFrame.from_dict(results).round(2)
    results.index = envs 
    # print(results)
    # results.to_csv(f'results/benchmarks_{lots}.csv')
    # Plot histogram of drifts for noise and flow
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data_for_plot['noise'], fill=False, label='Noise')
    sns.kdeplot(data_for_plot['flow'], fill=False, label='Flow')
    sns.kdeplot(data_for_plot['strategic'], fill=False, label='Strategic')
    plt.legend()
    plt.grid(True)
    plt.xlabel('mid price drift')
    plt.tight_layout()
    plt.xlim(-10, 10)
    plt.savefig('histogram.pdf')



