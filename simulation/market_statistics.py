# TODO: should integrate this whole thing into market gym directly

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
import sys
import matplotlib.pyplot as plt
import os 
from limit_order_book.plotting import heat_map

"""
    - this code is desgigned to obtain market statistic: like mid price drift, number of trades, number of events
    - the statistics are averaged over episodes 
    - the prioity queue is set up manually (no universal class for this, just do case by case)
    - save plots in the plots folder and trades in the results folder
    - TODO: could write the code using the Market environment class: need to modify the Market class 
        - currently, the market class always requires an execution agent
        - would need to have the case with and without execution agent
"""


class Market():
    def __init__(self, market_env='noise', seed=0, level=10):
        self.level = level 
        assert market_env in ['noise', 'flow', 'strategic']
        self.agents = {}

        time_delta = 15 
        # terminal_time = 300
        terminal_time = 600

        # override initial le1vels config 
        initial_agent_config['n_initial_levels'] = self.level
        noise_agent_config['level'] = self.level

        # initial agent setting 
        if market_env == 'noise':
            # initial_agent_config['initial_shape_file'] = 'initial_shape/noise_65.npz'
            # new initial shape file ! 
            initial_agent_config['initial_shape_file'] = 'noise_new_config.npz'            
            initial_agent_config['start_time'] = -time_delta
            agent = InitialAgent(**initial_agent_config)
            self.agents[agent.agent_id] = agent
        else:
            initial_agent_config['initial_shape_file'] = 'initial_shape/noise_flow_65.npz'
            initial_agent_config['start_time'] = -time_delta
            agent = InitialAgent(**initial_agent_config)
            self.agents[agent.agent_id] = agent
        agent = InitialAgent(**initial_agent_config)
        self.agents[agent.agent_id] = agent

        # noise agent setting
        noise_agent_config['rng'] = np.random.default_rng(seed)
        noise_agent_config['unit_volume'] = False
        noise_agent_config['terminal_time'] = terminal_time
        noise_agent_config['start_time'] = 0
        # noise_agent_config['fall_back_volume'] = 5

        # noise or flow 
        if market_env == 'noise':
            noise_agent_config['imbalance_reaction'] = False
            agent = NoiseAgent(**noise_agent_config)
        else: 
            noise_agent_config['imbalance_reaction'] = True
            agent = NoiseAgent(**noise_agent_config)
            # noise_agent_config['initial_shape_file'] = 'initial_shape/noise_flow_75_unit.npz'
            agent.limit_intensities = agent.limit_intensities * 0.85
            agent.market_intensity = agent.market_intensity * 0.85
            agent.cancel_intensities = agent.cancel_intensities * 0.85
        self.agents[agent.agent_id] = agent
        
        # strategic agent setting 
        if market_env == 'strategic':
            strategic_agent_config['time_delta'] = 3
            strategic_agent_config['market_volume'] = 1
            strategic_agent_config['limit_volume'] = 2
            strategic_agent_config['rng'] = np.random.default_rng(seed)
            strategic_agent_config['terminal_time'] = terminal_time
            strategic_agent_config['start_time'] = 0
            agent = StrategicAgent(**strategic_agent_config)
            self.agents[agent.agent_id] = agent 
        
    
    def reset(self):
        list_of_agents = list(self.agents.keys()) 
        # set the right level !!
        self.lob = LimitOrderBook(list_of_agents=list_of_agents, level=self.level, only_volumes=False)
        for agent_id in list_of_agents:
            self.agents[agent_id].reset()            
        # if 'strategic_agent' in self.agents:
        #     self.agents['strategic_agent'].direction = 'sell'
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

        trades = np.sum(self.lob.data.market_buy)+np.sum(self.lob.data.market_sell)

        buy_orders = np.sum(self.lob.data.market_buy)

        sell_orders = np.sum(self.lob.data.market_sell)

        return time, n_events, drift, trades, buy_orders, sell_orders


def rollout(seed, num_episodes, market_type, level=10):
    M = Market(market_env=market_type, seed=seed, level=level)
    n_events = []
    drifts = []
    times = []
    trades = []
    buy_orders = []
    sell_orders = []
    for _ in range(num_episodes):
        M.reset()
        time, n_event, drift, trade, buys, sells = M.run()
        n_events.append(n_event)
        drifts.append(drift)
        times.append(time)
        trades.append(trade)
        buy_orders.append(buys)
        sell_orders.append(sells)
    return n_events, drifts, times, trades, buy_orders, sell_orders


def mp_rollout(n_samples, n_cpus, market_type):
    """
    """
    samples_per_env = int(n_samples/n_cpus) 
    with Pool(n_cpus) as p:
        out = p.starmap(rollout, [(seed, samples_per_env, market_type) for seed in range(n_cpus)])    
    n_events, drifts, times, trades, buys, sells  = zip(*out)
    n_events = list(itertools.chain.from_iterable(n_events))
    times = list(itertools.chain.from_iterable(times))
    drifts = list(itertools.chain.from_iterable(drifts))
    trades = list(itertools.chain.from_iterable(trades))
    buys = list(itertools.chain.from_iterable(buys))
    sells = list(itertools.chain.from_iterable(sells))
    return n_events, drifts, times, trades, buys, sells 


if __name__ == '__main__':
    # create heat map plot here 
    max_level = 10 
    market_type = 'noise'
    seed = 1 
    do_heat_map = False
    if do_heat_map:
        M = Market(market_env=market_type, seed=seed)
        M.reset()
        M.run()
        level2, orders, market_orders = M.lob.log_to_df()
        heat_map(trades=market_orders, level2=level2, event_times=level2.time, max_level=7, scale=1000, max_volume=40, width=0.75*16, height=0.75*9, ylim=[996,1003], xlim=[0, 300])
        plt.tight_layout()
        plt.savefig(f'plots/heat_map_seed_{seed}.pdf')

    # only using levels is a speed up 
    num_episodes = 1
    start_time = time.time()                
    # note that the level aspect has very limited impact on the speed of the simulation. maybe a one second difference 
    # 5 levels: 15 seconds 
    # 10 levels: 16 seconds 
    # note: not really a big difference
    out = rollout(seed=0, num_episodes=num_episodes, market_type='noise', level=5)
    end_time = time.time()
    print(f"Execution time for {num_episodes} episodes:", end_time - start_time, "seconds")
    # print(out)
    # out = mp_rollout(n_samples=100, n_cpus=10, market_type='flow')
    # print(out
    do_mp_rollout = True
    if do_mp_rollout:
        # envs = ['noise', 'flow', 'strategic']
        # envs = ['noise', 'flow']
        envs = ['noise']
        # n_samples = 1000
        # n_samples = 1000
        n_samples = 256
        # n_cpus = 80
        # n_cpus = 128
        # takes about 16 seconds with 64 CPUs 
        n_cpus = 64
        results = {f'n_events': [],'drift_mean': [], 'drift_std': [], 'trades': [], 'trades_std': [], 'buy_orders': [], 'sell_orders': []} 
        data_drifts = {}
        data_for_trade_plot = {}
        start_time = time.time()
        for env in envs:
            start_time = time.time()
            n_events, drifts, times, trades, buys, sells = mp_rollout(n_samples=n_samples, n_cpus=n_cpus, market_type=env)
            end_time = time.time()
            print(f"Execution time for {n_samples} in parallel with {n_cpus} CPUS:", end_time - start_time, "seconds")
            data_drifts[env] = drifts
            data_for_trade_plot[env] = trades
            results[f'n_events'].append(np.mean(n_events))
            results[f'drift_mean'].append(np.mean(drifts))
            results[f'drift_std'].append(np.std(drifts))
            results[f'trades'].append(np.mean(trades))
            results[f'trades_std'].append(np.std(trades))
            results[f'buy_orders'].append(np.mean(buys))
            results[f'sell_orders'].append(np.mean(sells))
        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution time:", execution_time, "seconds")

        # process results 
        results = pd.DataFrame.from_dict(results).round(2)
        results.index = envs 
        print(results)
        # results.to_csv(f'results/market_stats_std8.csv')

        # histogram of drifts 
        fig, ax = plt.subplots(figsize=(10, 6))
        # colors = ['blue', 'green']
        colors = ['blue']
        bins = np.arange(-4.25, 5.75, 0.5)
        ax.hist(data_drifts['noise'], bins, density=False, histtype='bar', color=colors, label=['Noise', 'Flow'], rwidth=0.8)
        ax.set_xticks(np.arange(-4, 4.5, 0.5))
        ax.tick_params(axis='x', labelsize=7)
        ax.legend(prop={'size': 10})
        ax.set_title('Mid Price Drift', fontsize=12)
        plt.grid(True)
        plt.xlim(-4, 4)
        # plt.tight_layout()
        plt.savefig('plots/mid_price_drift_new_config.pdf')
        plt.figure(figsize=(10, 6))


        # plot histogram of trades for each market type
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data_for_trade_plot['noise'], fill=False, label='Noise')
        # sns.kdeplot(data_for_trade_plot['flow'], fill=False, label='Flow',  clip=(0, 150))
        plt.legend(fontsize=12)
        # plt.xlim(0, 140)
        plt.grid(True)
        plt.title('Number of Trades', fontsize=16)
        plt.ylabel('Frequency')
        # plt.tight_layout()
        plt.savefig('plots/kde_trades_std2_150.pdf')