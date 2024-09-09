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



def get_statistics(n_steps=1, rng=default_rng(0), initial_shape=50, damping_factor=1, 
                  imbalance=False, imbalance_factor=3, shape_file=None, frequency=1000, totol_trades_window=1000):
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
    # not for n_steps = 100 the order book will record 99 events because the initial event and the first noise agent event happen at the same time at t=0     
    for _ in range(n_steps):
        time, prio, agent_id = pq.get()
        orders = agents[agent_id].generate_order(lob=LOB, time=time) 
        # order = orders[0]
        # if order.volume is None:
        #     raise ValueError('Volume is None')        
        LOB.process_order_list(orders)
        out = agents[agent_id].new_event(time, agent_id)
        if out is not None:
            pq.put(out)

    # get bid and ask volumes but at a lower frequency. start from T/2 (by this time the order book should become more stable)
    T = len(LOB.data.bid_volumes)
    bid_volumes = LOB.data.bid_volumes[-int(T/2):][::frequency]
    ask_volumes = LOB.data.ask_volumes[-int(T/2):][::frequency]
    bestb = np.array(LOB.data.best_ask_prices[-int(T/2):][::frequency]) 
    besta = np.array(LOB.data.best_bid_prices[-int(T/2):][::frequency])
    # bes bid and ask volumes 
    # bestbv = np.array(LOB.data.best_bid_volumes[-int(T/2):][::100])
    # bestav = np.array(LOB.data.best_ask_volumes[-int(T/2):][::100])
    # mid proices 
    midp = (bestb + besta)/2
    midp_diff = np.diff(midp)
    midp = (np.array(LOB.data.best_ask_prices)+np.array(LOB.data.best_bid_prices))/2
    # totol trades over a sliding window of size totol_trades_window
    total_trades = np.array(LOB.data.market_buy[-int(T/2):])+np.array(LOB.data.market_sell[-int(T/2):])    
    window = np.lib.stride_tricks.sliding_window_view(total_trades, window_shape=totol_trades_window)
    total_trades = np.sum(window, axis=-1)
    average_time_step = np.mean(np.diff(LOB.data.time_stamps[-int(T/2):]))
    return bid_volumes, ask_volumes, midp_diff, midp, window.sum(axis=-1), average_time_step

def mp_rollout(n_samples, n_cpus, initial_shape, damping_factor, imbalance, frequency, total_trades_window, imbalance_factor=3):
    samples_per_cpu = int(n_samples/n_cpus)
    with Pool(n_cpus) as p:
        out = p.starmap(get_statistics, [(samples_per_cpu, default_rng(seed), initial_shape, damping_factor, imbalance, imbalance_factor, None, frequency, total_trades_window) for seed in range(n_cpus)])    
    bid_volumes, ask_volumes, midp_diff, midp, trades, average_time_step = zip(*out)
    bid_volumes = list(itertools.chain.from_iterable(bid_volumes))
    ask_volumes = list(itertools.chain.from_iterable(ask_volumes))
    midp_diff = list(itertools.chain.from_iterable(midp_diff))
    trades = list(itertools.chain.from_iterable(trades))
    average_time_step = list(average_time_step)
    return bid_volumes, ask_volumes, midp_diff, trades, average_time_step

def plot_prices(n_steps=int(1000), rng=default_rng(0), initial_shape=1, damping_factor=0.5, shape_file=None):
    bidv, askv, midp_diff, midp, trades, average_time_step = get_statistics(n_steps=int(n_steps), rng=rng, initial_shape=initial_shape, damping_factor=damping_factor, imbalance=False, shape_file=shape_file)
    bidv_imb, askv_imb, midp_diff_imb, midp_imb, trades_imb, average_time_step_imb = get_statistics(n_steps=int(n_steps), rng=rng, initial_shape=initial_shape, damping_factor=damping_factor, imbalance=True, shape_file=shape_file)
    plt.figure(figsize=(10, 6))
    plt.xlim(0, len(midp))
    plt.grid(True)
    plt.plot(midp, label='Noise')
    plt.plot(midp_imb, label='Flow')
    plt.legend()
    plt.savefig('plots/midp.pdf', dpi=350)
    print('price plot is done')

def trades_hist(trades, trades_imb):
    # TODO what does KDE plots actually do ? 
    # bar plot ? 
    plt.figure(figsize=(10, 6))
    sns.kdeplot(trades, fill=False, label=f'Noise')
    sns.kdeplot(trades_imb, fill=False, label=f'Noise+Flow,d={3}')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Number of Trades, Unit Volumes', fontsize=16)
    plt.tight_layout()
    plt.savefig('plots/trades_histogram.pdf', dpi=350)

def plot_average_shape(name, bidv, askv, bidv_imb, askv_imb, level=10):
    fig, axs = plt.subplots(figsize=(10, 6))
    plot_average_book_shape(bidv, askv, level=level, file_name=f'noise', title='noise', ax=axs)
    fig.tight_layout()
    fig.savefig(f'plots/average_shape_noise_{name}.pdf', dpi=350)
    # 
    fig, axs = plt.subplots(figsize=(10, 6))
    plot_average_book_shape(bidv_imb, askv_imb, level=10, file_name=f'noise_flow', title=f'noise+flow', ax=axs)        
    fig.tight_layout()
    fig.savefig(f'plots/average_shape_flow_{name}.pdf', dpi=350)
    # 
    print('saved average shape plots')

def plot_mid_price_changes(name, midp_diff, midp_diff_imb):
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
    plt.savefig(f'plots/midpn_changes_histogram_{name}.pdf', dpi=350)
    

if __name__ == '__main__':
    # compute average statistics 
    # bid_volumes, ask_volumes, midp_diff, midp, trades, average_time_step = get_statistics(n_steps=int(1e2), rng=default_rng(0), initial_shape=5, damping_factor=0.5, imbalance=False, frequency=10, totol_trades_window=10)
    # compute average statistics using multiprocessing
    start_time = timeit.default_timer()
    bidv, askv, midp_diff, trades, average_time_step = mp_rollout(n_samples=int(1e6), n_cpus=80, initial_shape=1, damping_factor=0.85, imbalance=False, frequency=100, total_trades_window=100)
    # np.savez('initial_shape/noise.npz', bidv=np.nanmean(bidv, axis=0), askv=np.nanmean(askv, axis=0))
    bidv_imb, askv_imb, midp_diff_imb, trades_imb, average_time_step_imb = mp_rollout(n_samples=int(1e6), n_cpus=60, initial_shape=1, damping_factor=0.75, imbalance=True, imbalance_factor=2, frequency=100, total_trades_window=100)
    # np.savez('initial_shape/noise_flow_75.npz', bidv=np.nanmean(bidv, axis=0), askv=np.nanmean(askv, axis=0))
    end_time = timeit.default_timer()
    print(f"Execution time: {end_time - start_time} seconds")
    # plot one price trajectory 
    # plot_prices(n_steps=int(1e5), rng=default_rng(4), initial_shape=5, damping_factor=0.8, shape_file='initial_shape/noise_unit.npz')
    # average_shape(n_steps=4000, rng=default_rng(0), initial_shape=5, damping_factor=0.5, imbalance=False)
    # print(f"average time step noise = {np.mean(average_time_step)}")
    # print(f"average time step noise+flow = {np.mean(average_time_step_imb)}")
    # #####    
    name = 'std2'
    plot_average_shape(name, bidv, askv, bidv_imb, askv_imb, level=30)
    # maybe bar plots make more sense here ?? 
    # trades_hist(trades, trades_imb)
    plot_mid_price_changes(name=name, midp_diff=midp_diff, midp_diff_imb=midp_diff_imb)