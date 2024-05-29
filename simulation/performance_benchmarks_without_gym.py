from agents import NoiseAgent, LinearSubmitLeaveAgent, StrategicAgent, SubmitAndLeaveAgent, MarketAgent, RLAgent
from limit_order_book.limit_order_book import LimitOrderBook
from config.config import noise_agent_config, strategic_agent_config, sl_agent_config, linear_sl_agent_config, market_agent_config
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
        assert market_env in ['noise', 'flow']

        # noise agent setting
        noise_agent_config['rng'] = np.random.default_rng(seed)
        noise_agent_config['unit_volume'] = False
        if market_env == 'noise':
            noise_agent_config['imbalance_reaction'] = False
            noise_agent_config['initial_shape_file'] = 'initial_shape/noise_unit.npz'
            self.noise_agent = NoiseAgent(**noise_agent_config)
        else: 
            noise_agent_config['imbalance_reaction'] = True
            noise_agent_config['imbalance_factor'] = 2.0
            noise_agent_config['damping_factor'] = 0.25
            # noise_agent_config['initial_shape_file'] = 'initial_shape/noise_flow_75_unit.npz'
            noise_agent_config['initial_shape_file'] = 'initial_shape/noise_unit.npz'
            self.noise_agent = NoiseAgent(**noise_agent_config)            
            self.noise_agent.limit_intensities = self.noise_agent.limit_intensities * 0.85
            self.noise_agent.market_intensity = self.noise_agent.market_intensity * 0.85
            self.noise_agent.cancel_intensities = self.noise_agent.cancel_intensities * 0.85
        
    def reset(self):
        list_of_agents = [self.noise_agent.agent_id]
        self.lob = LimitOrderBook(list_of_agents=list_of_agents, level=30, only_volumes=False)
        orders = self.noise_agent.initialize(time=0)        
        self.lob.process_order_list(orders)
        # initialize event queue 
        self.pq = PriorityQueue()
        # noise
        out = self.noise_agent.initial_event(self.lob)
        self.pq.put(out)
        # strategic
        self.pq.put((150,0,'stop'))
        return None 
    
    def run(self):
        n_events = 0 
        n_cancellations = 0 
        n_limits = 0 
        n_markets = 0 
        initial_mid = (self.lob.get_best_price('bid')+self.lob.get_best_price('ask'))/2
        event = None
        while not event=='stop':
            n_events += 1
            time, _, event = self.pq.get()
            if event == 'noise_agent_action':
                orders = self.noise_agent.generate_order(self.lob, time=time)
                if orders[0].type == 'cancellation_by_price_volume':
                    n_cancellations += 1
                elif orders[0].type == 'limit':
                    n_limits += 1
                elif orders[0].type == 'market':
                    n_markets += 1
                else:
                    raise ValueError(f'type={orders[0].type} not recognized')
                self.lob.process_order_list(orders)
                out = self.noise_agent.new_event(time, event)
                self.pq.put(out)
            elif event == 'stop':
                pass 
            else:
                raise ValueError(f'event={event} not recognized')
        
        drift = (self.lob.get_best_price('bid')+self.lob.get_best_price('ask'))/2 - initial_mid

        return n_events, n_cancellations, n_limits, n_markets, drift  


def rollout(seed, num_episodes, market_type):
    M = Market(market_env=market_type, seed=seed)
    n_events = []
    n_limits = []
    n_markets = []
    n_cancellations = []
    drifts = []
    for _ in range(num_episodes):
        M.reset()
        n_event, n_cancel, n_limit, n_market, drift = M.run()
        n_events.append(n_event)
        n_limits.append(n_limit)
        n_markets.append(n_market)
        n_cancellations.append(n_cancel)
        drifts.append(drift)
    return n_events, n_cancellations, n_limits, n_markets, drifts

def mp_rollout(n_samples, n_cpus, market_type):
    samples_per_env = int(n_samples/n_cpus) 
    with Pool(n_cpus) as p:
        out = p.starmap(rollout, [(seed, samples_per_env, market_type) for seed in range(n_cpus)])    
    n_events, n_cancel, n_limit, n_market, drift  = zip(*out)
    n_events = list(itertools.chain.from_iterable(n_events))
    n_cancel = list(itertools.chain.from_iterable(n_cancel))
    n_limit = list(itertools.chain.from_iterable(n_limit))
    n_market = list(itertools.chain.from_iterable(n_market))
    drift = list(itertools.chain.from_iterable(drift))
    return n_events, n_cancel, n_limit, n_market, drift 


# M = Market(market_env='flow', seed=1)
# M.reset()
# out = M.run()

# out = rollout(0, 10, 'flow') 
# print(out)

if __name__ == '__main__':
    n_samples = 1000
    n_cpus = 50
    results = {}
    results[f'n_events'] = []
    results[f'n_cancellations'] = []
    results[f'n_limits'] = []
    results[f'n_markets'] = []
    results[f'drift_mean'] = []
    results[f'drift_std'] = []
    data_for_plot = {}
    for env in ['noise', 'flow']:
        n_events, n_cancel, n_limit, n_market, drift = mp_rollout(n_samples, n_cpus, env)
        data_for_plot[env] = drift
        results[f'n_events'].append(np.mean(n_events))
        results[f'n_cancellations'].append(np.mean(n_cancel))
        results[f'n_limits'].append(np.mean(n_limit))
        results[f'n_markets'].append(np.mean(n_market))
        results[f'drift_mean'].append(np.mean(drift))
        results[f'drift_std'].append(np.std(drift))
    results = pd.DataFrame.from_dict(results).round(2)
    results.index = ['noise', 'flow']
    print(results)
    # results.to_csv(f'results/benchmarks_{lots}.csv')
    # Plot histogram of drifts for noise and flow
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data_for_plot['noise'], fill=False, label='Noise')
    sns.kdeplot(data_for_plot['flow'], fill=False, label='Noise+Flow')
    plt.legend()
    plt.grid(True)
    plt.xlabel('mid price drift')
    plt.tight_layout()
    plt.xlim(-10, 10)
    plt.savefig('histogram.pdf')



#     priority: int
#     item: Any = field(compare=False)

# Create a list to be used as a priority queue
# pq = PriorityQueue()
# Add some items to the priority queue
# pq.put(0, PrioritizedItem(2, "Clean the house"))
# pq.put(0, PrioritizedItem(1, "Write code"))
# pq.put(0, PrioritizedItem(3, "Read a book"))

# while not pq.empty():
#     out = pq.get()
#     print(out)

# initialize agents and limit order book 

if False:
    noise_agent_config['initial_shape_file'] = 'initial_shape/noise_unit.npz'
    noise_agent_config['rng'] = np.random.default_rng(4)
    noise_agent_config['start_time'] = 0 
    NA = NoiseAgent(**noise_agent_config)
    EA = MarketAgent(start_time=0, volume=50)
    SA = StrategicAgent(start_time=0, time_delta=50, market_volume=1, limit_volume=1, rng=np.random.default_rng(0))
    SA.reset_direction()
    print(SA.direction)
    # this reset
    LOB = LimitOrderBook(list_of_agents=[NA.agent_id, EA.agent_id, SA.agent_id], level=30, only_volumes=False)


    # initialize LOB 
    orders = NA.initialize(time=0)
    LOB.process_order_list(orders)
    # print(LOB.level2('bid'))
    # print(LOB.level2('ask'))
    # print(LOB.order_map)

    # priority queues work like this 
    pq = PriorityQueue()

    # pq.put((0, 1,'task 1'))
    # pq.put((0, 0, 'task 2'))
    # pq.put((1, 2, 'task 3')) 
    # while not pq.empty():
    #     priority, _, task = pq.get()
    #     print(f"Processing {task} with priority {priority}")


    # initialize event queue 
    # execution 
    out = EA.initial_event()
    pq.put(out)
    # noise, first event  
    out = NA.initial_event(LOB)
    pq.put(out)
    # strategic. first event  
    out = SA.initial_event()
    pq.put(out)

    # while not pq.empty():
    #     t, p, event = pq.get()
    #     print(f"event={event}, time={t}, p={p}")


    terminated = False
    observation = False
    while not terminated and not observation: 
        time, _, event = pq.get()
        print(f'{event}, time={time}')
        if event == 'execution_agent_action':
            orders = EA.generate_order(time, LOB)
            msgs = LOB.process_order_list(orders)
            rewards, terminated = EA.update_position_from_message_list(msgs)
            if terminated:
                break
            else:            
                out = EA.new_event(time, event)
                pq.put(out)
        elif event == 'execution_agent_observation':
            out = EA.new_event(time, event)
            pq.put(out)
            observation = True
            break 
        elif event == 'noise_agent_action':
            orders = NA.generate_order(LOB, time=time)
            msgs = LOB.process_order_list(orders)
            rewards, terminated = EA.update_position_from_message_list(msgs)
            if terminated:
                break
            else:
                out = NA.new_event(time, event)
                pq.put(out)
        elif event == 'strategic_agent_action':
            orders = SA.generate_order(LOB, time=time)
            msgs = LOB.process_order_list(orders)
            rewards, terminated = EA.update_position_from_message_list(msgs)
            if terminated:
                break
            else:
                # must run .generate_order() before running .new_event() ! 
                out = SA.new_event(time, event)
                pq.put(out)
        else:
            raise ValueError(f'event={event} not recognized')

    print(f'total reward: {EA.cummulative_reward}')
    print(f"bid price: {LOB.get_best_price('bid')}")
