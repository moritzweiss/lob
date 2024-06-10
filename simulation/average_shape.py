import os 
import sys 
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)
from config.config import noise_agent_config
from limit_order_book.limit_order_book import LimitOrderBook
from simulation.agents import NoiseAgent, InitialAgent
import numpy as np
from numpy.random import default_rng
import time 
import timeit
from limit_order_book.plotting import plot_average_book_shape
import matplotlib.pyplot as plt
from multiprocessing import Pool
import itertools
import seaborn as sns
from config.config import noise_agent_config, initial_agent_config
from queue import PriorityQueue



def average_shape(n_steps=1, rng=default_rng(0), initial_shape=50, damping_factor=1, imbalance=False, imbalance_factor=3, shape_file=None):
    agents = {}

    # noise agent 
    noise_agent_config['damping_factor'] = damping_factor
    noise_agent_config['imbalance_reaction'] = imbalance
    noise_agent_config['imbalance_factor'] = imbalance_factor
    noise_agent_config['rng'] = rng    
    noise_agent_config['unit_volume'] = False
    noise_agent_config['terminal_time'] = np.inf 
    noise_agent_config['fall_back_volume'] = initial_shape
    if shape_file is not None:
        noise_agent_config['initial_shape_file'] = shape_file
    agent = NoiseAgent(**noise_agent_config)
    agents[agent.agent_id] = agent

    # initial agent
    initial_agent_config['initial_shape'] = initial_shape
    agent = InitialAgent(**initial_agent_config)
    agents[agent.agent_id] = agent

    # LOB 
    list_of_agents = [agent.agent_id for agent in agents.values()]
    LOB = LimitOrderBook(list_of_agents=list_of_agents, level=30, only_volumes=True)

    # reset agents 
    for agent in agents.values():
        agent.reset()

    # pq 
    pq = PriorityQueue()
    for agent in agents.values():
        out = agent.initial_event()
        pq.put(out)

    # simulation
    for _ in range(n_steps):
        time, prio, agent_id = pq.get()
        orders = agents[agent_id].generate_order(lob=LOB, time=time) 
        order = orders[0]
        if order.volume is None:
            raise ValueError('Volume is None')        
        LOB.process_order_list(orders)
        out = agents[agent_id].new_event(time, agent_id)
        if out is not None:
            pq.put(out)
    # 
    T = len(LOB.data.bid_volumes)
    bid_volumes = LOB.data.bid_volumes[-int(T/2):][::1000]
    ask_volumes = LOB.data.ask_volumes[-int(T/2):][::1000]
    # bid_volumes = LOB.data.bid_volumes
    # ask_volumes = LOB.data.ask_volumes
    # bestb = np.array(LOB.data.best_bid_prices[-int(T/2):][::100])
    # besta = np.array(LOB.data.best_ask_prices[-int(T/2):][::100])
    bestb = np.array(LOB.data.best_ask_prices[-int(T/2):][::1000]) 
    besta = np.array(LOB.data.best_bid_prices[-int(T/2):][::1000])
    # bestbv = np.array(LOB.data.best_bid_volumes[-int(T/2):][::100])
    # bestav = np.array(LOB.data.best_ask_volumes[-int(T/2):][::100])
    midp = (bestb + besta)/2
    midp_diff = np.diff(midp)
    midp = (np.array(LOB.data.best_ask_prices)+np.array(LOB.data.best_bid_prices))/2
    # 
    total_trades = np.array(LOB.data.market_buy[-int(T/2):])+np.array(LOB.data.market_sell[-int(T/2):])    
    window = np.lib.stride_tricks.sliding_window_view(total_trades, window_shape=1000)
    total_trades = np.sum(window, axis=-1)
    average_time_step = np.mean(np.diff(LOB.data.time_stamps[-int(T/2):]))
    return bid_volumes, ask_volumes, midp_diff, midp, window.sum(axis=-1), average_time_step

def mp_rollout(n_samples, n_cpus, initial_shape, damping_factor, imbalance, imbalance_factor=3):
    samples_per_cpu = int(n_samples/n_cpus)
    with Pool(n_cpus) as p:
        out = p.starmap(average_shape, [(samples_per_cpu, default_rng(seed), initial_shape, damping_factor, imbalance, imbalance_factor) for seed in range(n_cpus)])    
    bid_volumes, ask_volumes, midp_diff, midp, trades, average_time_step = zip(*out)
    bid_volumes = list(itertools.chain.from_iterable(bid_volumes))
    ask_volumes = list(itertools.chain.from_iterable(ask_volumes))
    midp_diff = list(itertools.chain.from_iterable(midp_diff))
    trades = list(itertools.chain.from_iterable(trades))
    average_time_step = list(average_time_step)
    return bid_volumes, ask_volumes, midp_diff, trades, average_time_step

def plot_prices(n_steps=int(1000), rng=default_rng(0), initial_shape=1, damping_factor=0.5, shape_file=None):
    bidv, askv, midp_diff, midp, trades, average_time_step = average_shape(n_steps=int(n_steps), rng=rng, initial_shape=initial_shape, damping_factor=damping_factor, imbalance=False, shape_file=shape_file)
    bidv_imb, askv_imb, midp_diff_imb, midp_imb, trades_imb, average_time_step_imb = average_shape(n_steps=int(n_steps), rng=rng, initial_shape=initial_shape, damping_factor=damping_factor, imbalance=True, shape_file=shape_file)
    plt.figure(figsize=(10, 6))
    plt.xlim(0, len(midp))
    plt.grid(True)
    plt.plot(midp, label='Noise')
    plt.plot(midp_imb, label='Flow')
    plt.legend()
    plt.savefig('plots/midp.pdf', dpi=350)
    print('price plot is done')

def trades_hist(trades, trades_imb):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(trades, fill=False, label=f'Noise')
    sns.kdeplot(trades_imb, fill=False, label=f'Noise+Flow,d={3}')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Number of Trades, Unit Volumes', fontsize=16)
    plt.tight_layout()
    plt.savefig('plots/trades_histogram.pdf', dpi=350)

def plot_average_shape(bidv, askv, bidv_imb, askv_imb, level=10):
    fig, axs = plt.subplots(figsize=(10, 6))
    plot_average_book_shape(bidv, askv, level=level, file_name=f'noise', title='noise', ax=axs)
    fig.tight_layout()
    fig.savefig('plots/average_shape_noise.pdf', dpi=350)
    # 
    fig, axs = plt.subplots(figsize=(10, 6))
    plot_average_book_shape(bidv_imb, askv_imb, level=10, file_name=f'noise_flow', title=f'noise+flow', ax=axs)        
    fig.tight_layout()
    fig.savefig('plots/average_shape_flow.pdf', dpi=350)
    # 
    print('saved average shape plots')

def plot_mid_price_changes(midp_diff, midp_diff_imb):
    plt.figure(figsize=(10, 6))
    # print(midp_diff_imb)
    sns.kdeplot(midp_diff, fill=False, label='Noise')
    sns.kdeplot(midp_diff_imb, fill=False, label=f'Noise+Flow')
    # sns.kdeplot(midp_diff_imb_7, fill=False, label=f'Noise+Flow,c={0.75}')
    # sns.kdeplot(midp_diff_imb_5, fill=False, label=f'Noise+Flow,c={0.5}')
    # plt.hist(midp1_changes, bins=np.arange(-20,21,1), edgecolor='black', alpha=0.5, label='Damping Factor = 1')
    plt.xlabel('Change in Mid Price')
    plt.ylabel('Frequency')
    plt.title('Histogram of Mid Price Changes')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.xlim(-10, 10)
    plt.savefig('plots/midpn_changes_histogram.pdf', dpi=350)
    

if __name__ == '__main__':
    # plot_prices(n_steps=int(1e5), rng=default_rng(4), initial_shape=5, damping_factor=0.2, shape_file='initial_shape/noise_unit.npz')
    # average_shape(n_steps=4000, rng=default_rng(0), initial_shape=5, damping_factor=0.5, imbalance=False)
    ####
    N = int(1e6)
    start_time = timeit.default_timer()
    bidv, askv, midp_diff, trades, average_time_step = mp_rollout(n_samples=N, n_cpus=60, initial_shape=1, damping_factor=0, imbalance=False)
    np.savez('initial_shape/noise.npz', bidv=np.nanmean(bidv, axis=0), askv=np.nanmean(askv, axis=0))
    bidv_imb, askv_imb, midp_diff_imb, trades_imb, average_time_step_imb = mp_rollout(n_samples=N, n_cpus=60, initial_shape=1, damping_factor=0.75, imbalance=True, imbalance_factor=2)
    np.savez('initial_shape/noise_flow_75.npz', bidv=np.nanmean(bidv, axis=0), askv=np.nanmean(askv, axis=0))
    end_time = timeit.default_timer()
    # print(f"average time step noise = {np.mean(average_time_step)}")
    # print(f"average time step noise+flow = {np.mean(average_time_step_imb)}")
    # print(f"Execution time: {end_time - start_time} seconds")
    # #####    
    plot_average_shape(bidv, askv, bidv_imb, askv_imb, level=30)
    # trades_hist(trades, trades_imb)
    # plot_mid_price_changes(midp_diff=midp_diff, midp_diff_imb=midp_diff_imb)