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


def average_shape(n_time_steps=1, rng=default_rng(0), initial_shape=50, damping_factor=1, imbalance=False):
    noise_agent_config['initial_shape'] = initial_shape
    noise_agent_config['damping_factor'] = damping_factor
    noise_agent_config['imbalance_reaction'] = imbalance
    noise_agent_config['rng'] = rng    
    noise_agent_config['unit_volume'] = False
    noise_agent_config['market_std'] = 2
    noise_agent_config['limit_std'] = 1
    noise_agent_config['cancel_std'] = 1
    noise_agent_config['imbalance_factor'] = 3
    NA = NoiseAgent(**noise_agent_config)
    LOB = LimitOrderBook(list_of_agents=[NA.agent_id], level=30, only_volumes=True)
    orders = NA.initialize(time=0)
    LOB.process_order_list(orders)
    for time in range(n_time_steps):
        order, _ = NA.sample_order(LOB, time=time)
        LOB.process_order(order)
        LOB.clear_orders(30)
    T = len(LOB.data.bid_volumes)
    bid_volumes = LOB.data.bid_volumes[-int(T/2):][::100]
    ask_volumes = LOB.data.ask_volumes[-int(T/2):][::100]     
    # bestb = np.array(LOB.data.best_bid_prices[-int(T/2):][::100])
    # besta = np.array(LOB.data.best_ask_prices[-int(T/2):][::100])
    bestb = np.array(LOB.data.best_ask_prices[-int(T/2):]) 
    besta = np.array(LOB.data.best_bid_prices[-int(T/2):])
    bestbv = np.array(LOB.data.best_bid_volumes[-int(T/2):])
    bestav = np.array(LOB.data.best_ask_volumes[-int(T/2):])
    midp = (bestb + besta)/2    
    microp = (bestav*bestb + bestbv*besta)/(bestav+bestbv)
    return bid_volumes, ask_volumes, midp, microp 

def mp_rollout(n_samples, n_cpus, initial_shape, damping_factor, imbalance):
    samples_per_cpu = int(n_samples/n_cpus)
    with Pool(n_cpus) as p:
        out = p.starmap(average_shape, [(samples_per_cpu, default_rng(seed), initial_shape, damping_factor, imbalance) for seed in range(n_cpus)])    
    bid_volumes, ask_volumes, midp, microp = zip(*out)
    bid_volumes = list(itertools.chain.from_iterable(bid_volumes))
    ask_volumes = list(itertools.chain.from_iterable(ask_volumes))
    midp = list(itertools.chain.from_iterable(midp))
    microp = list(itertools.chain.from_iterable(microp))
    return bid_volumes, ask_volumes, midp, microp

if __name__ == '__main__':
    N = int(1e6)
    start_time = timeit.default_timer()
    # average_shape(n_time_steps=int(N), rng=default_rng(0), initial_shape=1, damping_factor=1, imbalance=True)
    # bidv0, askv0, midp0, mp = mp_rollout(N, 50, 1, 0, True)
    # bidv5, askv5, midp5 = mp_rollout(N, 50, 1, 0.5, True)
    bidv1, askv1, midp1, mp1 = mp_rollout(N, 50, 1, 1, True)
    bidvn, askvn, midpn, mpn = mp_rollout(N, 50, 1, 0, False)
    if True:
        # bidv, askv, midp = average_shape(n_time_steps=int(1e4), rng=defƒault_rng(0), initial_shape=1)
        end_time = timeit.default_timer()
        print(f"Execution time: {end_time - start_time} seconds")
        fig, axs = plt.subplots(2)
        plot_average_book_shape(bidvn, askvn, level=10, file_name=f'shape_c=0_no_imb', title='no imb reaction', ax=axs[0])
        plot_average_book_shape(bidv1, askv1, level=10, file_name=f'shape_c=1', title='c=1', ax=axs[1])
        # plot_average_book_shape(bidv5, askv5, level=10, file_name=f'shape_c=0.5', title='c=0.5', ax=axs[2])
        # plot_average_book_shape(bidv0, askv0, level=10, file_name=f'shape_c=0', title='c=0', ax=axs[3])
        axs[1].set_xlabel('distance to mid')
        fig.tight_layout()
        fig.savefig('average_shape.pdf', dpi=350)
        # plt.figure()
        # plot_average_book_shape(bidv, askv, level=10, file_name=f'shape_c=1', title='c=1')
        # plt.figure()
        # plot_average_book_shape(bidv0, askv0, level=10, file_name=f'shape_c=0', title='c=0')
        # plt.figure()
        # plot_average_book_shape(bidv5, askv5, level=10, file_name=f'shape_c=0.5', title='c=no imb reaction')
    if True:
        N = int(2e3)        
        plt.figure(figsize=(10, 6))
        midpn_changes = np.diff(midpn[::1000])
        midp1_changes = np.diff(midp1[::1000])
        plt.hist(midpn_changes, bins=np.arange(-20,21,1), edgecolor='black', alpha=0.5, label='No Imbalance')
        plt.hist(midp1_changes, bins=np.arange(-20,21,1), edgecolor='black', alpha=0.5, label='Damping Factor = 1')
        plt.xlabel('Change in Mid Price')
        plt.ylabel('Frequency')
        plt.title('Histogram of Mid Price Changes')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.xlim(-20, 20)
        plt.savefig('midpn_changes_histogram.pdf', dpi=350)











