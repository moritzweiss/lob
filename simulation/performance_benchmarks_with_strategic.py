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


class Market():
    def __init__(self, market_env='noise', execution_agent='sl_agent', volume=10, seed=0):

        # noise agent setting
        noise_agent_config['rng'] = np.random.default_rng(seed)
        noise_agent_config['unit_volume'] = True
        if market_env == 'noise':
            noise_agent_config['imbalance_reaction'] = False
            noise_agent_config['initial_shape_file'] = 'initial_shape/noise_unit.npz'
            self.noise_agent = NoiseAgent(**noise_agent_config)
        else: 
            noise_agent_config['imbalance_reaction'] = True
            noise_agent_config['imbalance_factor'] = 2.0
            # noise_agent_config['initial_shape_file'] = 'initial_shape/noise_flow_75_unit.npz'
            noise_agent_config['initial_shape_file'] = 'initial_shape/noise_unit.npz'
            self.noise_agent = NoiseAgent(**noise_agent_config)            
            self.noise_agent.limit_intensities = self.noise_agent.limit_intensities * 0.7
            self.noise_agent.market_intensity = self.noise_agent.market_intensity * 0.7
            self.noise_agent.cancel_intensities = self.noise_agent.cancel_intensities * 0.7


        # strategic agent settings 
        strategic_agent_config['time_delta'] = 7.5
        strategic_agent_config['market_volume'] = 1
        strategic_agent_config['limit_volume'] = 1
        if market_env == 'strategic':
            self.stratetic_agent = StrategicAgent(**strategic_agent_config)
        else:
            self.stratetic_agent = None 

        # execution agent
        if execution_agent == 'market_agent':
            market_agent_config['volume'] = volume
            sl_agent_config['start_time'] = 0
            self.execution_agent = MarketAgent(**market_agent_config)
        elif execution_agent == 'sl_agent':
            sl_agent_config['volume'] = volume
            sl_agent_config['start_time'] = 0
            sl_agent_config['terminal_time'] = 150
            self.execution_agent = SubmitAndLeaveAgent(**sl_agent_config)
        elif execution_agent == 'linear_sl_agent':
            linear_sl_agent_config['start_time'] = 0
            linear_sl_agent_config['terminal_time'] = 150
            linear_sl_agent_config['time_delta'] = 15
            linear_sl_agent_config['volume'] = volume
            self.execution_agent = LinearSubmitLeaveAgent(**linear_sl_agent_config)
        else:
            raise ValueError(f'execution_agent={execution_agent} not recognized')
        

        
    def reset(self):
        if self.stratetic_agent is not None:
            self.stratetic_agent.reset_direction()
            list_of_agents = [self.noise_agent.agent_id, self.execution_agent.agent_id, self.stratetic_agent.agent_id]
        else:
            list_of_agents = [self.noise_agent.agent_id, self.execution_agent.agent_id]
        self.lob = LimitOrderBook(list_of_agents=list_of_agents, level=30, only_volumes=False)
        self.execution_agent.reset()
        orders = self.noise_agent.initialize(time=0)        
        self.lob.process_order_list(orders)
        # initialize event queue 
        self.pq = PriorityQueue()
        # execution 
        self.execution_agent.reset()
        out = self.execution_agent.initial_event()
        self.pq.put(out)
        # noise
        out = self.noise_agent.initial_event(self.lob)
        self.pq.put(out)
        # strategic
        if self.stratetic_agent is not None:
            out = self.stratetic_agent.initial_event()
            self.pq.put(out)
        return None 
    
    def run(self):
        n_events = 0 
        n_cancellations = 0 
        n_limits = 0 
        n_markets = 0 
        initial_bid = self.lob.get_best_price('bid')
        terminated = False
        observation = False
        while not terminated and not observation: 
            n_events += 1
            time, _, event = self.pq.get()
            # print(f'{event}, time={time}')
            if event == 'execution_agent_action':
                orders = self.execution_agent.generate_order(time, self.lob)
                msgs = self.lob.process_order_list(orders)
                rewards, terminated = self.execution_agent.update_position_from_message_list(msgs)
                if terminated:
                    break
                else:            
                    out = self.execution_agent.new_event(time, event)
                    self.pq.put(out)
            elif event == 'execution_agent_observation':
                out = self.execution_agent.new_event(time, event)
                self.pq.put(out)
                observation = True
                break 
            elif event == 'noise_agent_action':
                orders = self.noise_agent.generate_order(self.lob, time=time)
                if orders[0].type == 'cancellation_by_price_volume':
                    n_cancellations += 1
                elif orders[0].type == 'limit':
                    n_limits += 1
                elif orders[0].type == 'market':
                    n_markets += 1
                else:
                    raise ValueError(f'type={orders[0].type} not recognized')
                msgs = self.lob.process_order_list(orders)
                rewards, terminated = self.execution_agent.update_position_from_message_list(msgs)
                if terminated:
                    break
                else:
                    out = self.noise_agent.new_event(time, event)
                    self.pq.put(out)
            elif event == 'strategic_agent_action':
                orders = self.stratetic_agent.generate_order(self.lob, time=time)
                msgs = self.lob.process_order_list(orders)
                rewards, terminated = self.execution_agent.update_position_from_message_list(msgs)
                if terminated:
                    break
                else:
                    # must run .generate_order() before running .new_event() ! 
                    out = self.stratetic_agent.new_event(time, event)
                    self.pq.put(out)
            else:
                raise ValueError(f'event={event} not recognized')
        
        drift = self.lob.get_best_price('bid') - initial_bid

        return self.execution_agent.cummulative_reward, n_events, n_cancellations, n_limits, n_markets, drift  


def rollout(seed, num_episodes, execution_agent, market_type, volume):
    M = Market(volume=volume, execution_agent=execution_agent, market_env=market_type, seed=seed)
    total_rewards = []
    n_events = []
    n_limits = []
    n_markets = []
    n_cancellations = []
    drifts = []
    for _ in range(num_episodes):
        M.reset()
        total_reward, n_event, n_cancel, n_limit, n_market, drift = M.run()
        total_rewards.append(total_reward)
        n_events.append(n_event)
        n_limits.append(n_limit)
        n_markets.append(n_market)
        n_cancellations.append(n_cancel)
        drifts.append(drift)
    return total_rewards, n_events, n_cancellations, n_limits, n_markets, drifts

# note: can also use ray for multiprocessing rollouts 
def mp_rollout(n_samples, n_cpus, execution_agent, market_type, volume):
    samples_per_env = int(n_samples/n_cpus) 
    with Pool(n_cpus) as p:
        out = p.starmap(rollout, [(seed, samples_per_env, execution_agent, market_type, volume) for seed in range(n_cpus)])    
    all_rewards, n_events, n_cancel, n_limit, n_market, drift  = zip(*out)
    all_rewards = list(itertools.chain.from_iterable(all_rewards))
    n_events = list(itertools.chain.from_iterable(n_events))
    n_cancel = list(itertools.chain.from_iterable(n_cancel))
    n_limit = list(itertools.chain.from_iterable(n_limit))
    n_market = list(itertools.chain.from_iterable(n_market))
    drift = list(itertools.chain.from_iterable(drift))
    return all_rewards, n_events, n_cancel, n_limit, n_market, drift 


M = Market(volume=10, execution_agent='linear_sl_agent', market_env='flow', seed=1)
M.reset()
out = M.run()

# out = rollout(0, 10, 'linear_sl_agent', 'flow', 10)
# print(out)

if __name__ == '__main__':
    n_samples = 1000
    n_cpus = 50
    for lots in [10, 40]:
        results = {}
        for agent in ['sl_agent', 'linear_sl_agent']:
            results[f'{agent}_reward_mean'] = []
            results[f'{agent}_reward_std'] = []
            results[f'{agent}_n_events'] = []
            results[f'{agent}_n_cancellations'] = []
            results[f'{agent}_n_limits'] = []
            results[f'{agent}_n_markets'] = []
            results[f'{agent}_drift'] = []
            for env in ['noise', 'flow']:
                rewards, n_events, n_cancel, n_limit, n_market, drift = mp_rollout(n_samples, n_cpus, agent, env, lots)
                results[f'{agent}_reward_mean'].append(np.mean(rewards))
                results[f'{agent}_reward_std'].append(np.std(rewards))
                results[f'{agent}_n_events'].append(np.mean(n_events))
                results[f'{agent}_n_cancellations'].append(np.mean(n_cancel))
                results[f'{agent}_n_limits'].append(np.mean(n_limit))
                results[f'{agent}_n_markets'].append(np.mean(n_market))
                results[f'{agent}_drift'].append(np.mean(drift))
        results = pd.DataFrame.from_dict(results).round(2)
        results.index = ['noise', 'flow']
        print(results)
        results.to_csv(f'results/benchmarks_{lots}.csv')



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
