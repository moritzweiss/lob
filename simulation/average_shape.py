import os 
import sys 
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)
from config.config import config
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


def average_shape(n_time_steps=1, rng=default_rng(0)):
    NA = NoiseAgent(level=30, rng=rng, initial_shape=50, config_n=1, imbalance_reaction=True, damping_factor=0.5)
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
    return bid_volumes, ask_volumes

def mp_rollout(n_samples, n_cpus):
    samples_per_cpu = int(n_samples/n_cpus)
    with Pool(n_cpus) as p:
        out = p.starmap(average_shape, [(samples_per_cpu, default_rng(seed)) for seed in range(n_cpus)])    
    bid_volumes, ask_volumes = zip(*out)
    bid_volumes = list(itertools.chain.from_iterable(bid_volumes))
    ask_volumes = list(itertools.chain.from_iterable(ask_volumes))
    return bid_volumes, ask_volumes

if __name__ == '__main__':
    start_time = timeit.default_timer()
    bidv, askv = mp_rollout(int(2e6), 50)
    # bidv, askv = average_shape(n_time_steps=int(1e5), rng=default_rng(0))
    end_time = timeit.default_timer()
    print(f"Execution time: {end_time - start_time} seconds")
    plot_average_book_shape(bidv, askv)
    # plt.show()








