import os 
import sys 
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)
from config.config import noise_agent_config
from limit_order_book.limit_order_book import LimitOrderBook
from simulation.agents import NoiseAgent
import numpy as np
from numpy.random import default_rng
import time 
import timeit
from limit_order_book.plotting import plot_average_book_shape
import matplotlib.pyplot as plt
from multiprocessing import Pool
import itertools
import seaborn as sns
from config.config import noise_agent_config


def average_shape(n_steps=1, rng=default_rng(0), initial_shape=50, damping_factor=1, imbalance=False, imbalance_factor=3):
    noise_agent_config['initial_shape'] = initial_shape
    noise_agent_config['damping_factor'] = damping_factor
    noise_agent_config['imbalance_reaction'] = imbalance
    noise_agent_config['imbalance_factor'] = imbalance_factor
    noise_agent_config['rng'] = rng    
    noise_agent_config['unit_volume'] = False
    # noise_agent_config['initial_shape_file'] = 'initial_shape/noise_flow_75_unit.npz'
    NA = NoiseAgent(**noise_agent_config)
    LOB = LimitOrderBook(list_of_agents=[NA.agent_id], level=30, only_volumes=True)
    orders = NA.initialize(time=0)
    LOB.process_order_list(orders)
    t = 0
    for _ in range(n_steps):
        order = NA.generate_order(LOB, time=t)
        t += NA.waiting_time
        LOB.process_order_list(order)
        LOB.clear_orders(30)
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

if __name__ == '__main__':
    average_shape(n_steps=int(2e3), rng=default_rng(0), initial_shape=1, damping_factor=0.5, imbalance=True)
    # bidv0, askv0, midp0, mp = mp_rollout(N, 50, 1, 0, True)
    # bidv5, askv5, midp5 = mp_rollout(N, 50, 1, 0.5, True)
    if False: 
        N = int(1e5)
        bidv, askv, midp_diff, midp, _, _  = average_shape(n_steps=int(N), rng=default_rng(0), initial_shape=10, damping_factor=0.5, imbalance=False)
        damping_factor = 0.7
        bidv_imb, askv_imb, midp_diff_imb, midp_imb, _, _ = average_shape(n_steps=int(N), rng=default_rng(0), initial_shape=10, damping_factor=damping_factor, imbalance=True, imbalance_factor=3)
        # bidv_imb_5, askv_imb_5, midp_diff_imb_5, midp_imb_5, _ = average_shape(n_time_steps=int(N), rng=default_rng(0), initial_shape=1, damping_factor=0.5, imbalance=True)
        # bidv_imb_7, askv_imb_7, midp_diff_imb_7, midp_imb_7, _ = average_shape(n_time_steps=int(N), rng=default_rng(0), initial_shape=1, damping_factor=0.75, imbalance=True)
        plt.figure(figsize=(10, 6))
        plt.xlim(0, len(midp))
        plt.grid(True)
        plt.plot(midp, label='Noise')
        plt.plot(midp_imb, label=f'Noise+Flow, c={damping_factor}')
        # plt.plot(midp_imb_5, label='Noise+Flow, c=0.5')
        # plt.plot(midp_imb_7, label='Noise+Flow, c=0.75')
        plt.legend()
        plt.savefig('midp.pdf', dpi=350)
        print('DONE')                
    if True:
        N = int(1e6)
        # damping_factor = 0.5 
        start_time = timeit.default_timer()
        bidv, askv, midp_diff, trades, average_time_step = mp_rollout(N, 60, 1, 0, False)
        # print(np.nanmean(bidv, axis=0))
        np.savez('initial_shape/noise_unit.npz', bidv=np.nanmean(bidv, axis=0), askv=np.nanmean(askv, axis=0))
        bidv_imb, akv_imb, midp_diff_imb, trades_imb, average_time_step_imb = mp_rollout(N, 60, 1, damping_factor=0.7, imbalance=True)
        # bidv_imb_5, askv_imb_5, midp_diff_imb_5, trades_imb_5 = mp_rollout(N, 50, 10, damping_factor=0.5, imbalance=True)
        # bidv_imb_7, askv_imb_7, midp_diff_imb_7, trades_imb_7 = mp_rollout(N, 50, 10, damping_factor=0.75, imbalance=True)
        np.savez('initial_shape/noise_flow_75_unit.npz', bidv=np.nanmean(bidv, axis=0), askv=np.nanmean(askv, axis=0))
        end_time = timeit.default_timer()
        print(f"average time step noise = {np.mean(average_time_step)}")
        print(f"average time step noise+flow = {np.mean(average_time_step_imb)}")
        print(f"Execution time: {end_time - start_time} seconds")
    if True: 
        # bidv_imb, akv_imb, midp_diff_imb, trades_imb = mp_rollout(N, 50, 10, damping_factor=1.0, imbalance=False, imbalance_factor=1)
        # bidv_imb_5, askv_imb_5, midp_diff_imb_5, trades_imb_3 = mp_rollout(N, 50, 10, damping_factor=0.75, imbalance=True, imbalance_factor=3)
        # print(midp_diff_imb)
        plt.figure(figsize=(10, 6))
        sns.kdeplot(trades, fill=False, label=f'Noise')
        sns.kdeplot(trades_imb, fill=False, label=f'Noise+Flow,d={3}')
        plt.grid(True)
        plt.legend()
        plt.xlabel('Number of Trades, Unit Volumes', fontsize=16)
        plt.tight_layout()
        plt.savefig('trades_histogram.pdf', dpi=350)        
    if True:
        # bidv, askv, midp = average_shape(n_time_steps=int(1e4), rng=defƒault_rng(0), initial_shape=1)
        fig, axs = plt.subplots(figsize=(10, 6))
        plot_average_book_shape(bidv, askv, level=10, file_name=f'noise', title='noise', ax=axs)
        fig.tight_layout()
        fig.savefig('average_shape.pdf', dpi=350)
        # 
        fig, axs = plt.subplots(figsize=(10, 6))
        plot_average_book_shape(bidv_imb, akv_imb, level=10, file_name=f'noise_flow', title=f'noise+flow', ax=axs)        
        fig.tight_layout()
        fig.savefig('average_shape_imbalance.pdf', dpi=350)
    if False:
        N = int(2e3)        
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
        plt.savefig('midpn_changes_histogram.pdf', dpi=350)
    # if True: 
        # midpn_changes = np.diff(midpn)
        # midp1_changes = np.diff(midp1)        
        # large_changes = np.where(np.abs(midp1_changes) > 10)[0]
        # print(f"Number of large changes: {len(large_changes)}")
        # print("Indices of large changes:", large_changes)
        # print("Bid volume before change:", bidv1[large_changes[0]])
        # print("Ask volume before change:", askv1[large_changes[0]])        











