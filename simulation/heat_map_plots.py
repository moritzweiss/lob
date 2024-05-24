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
from limit_order_book.plotting import plot_average_book_shape, heat_map
import matplotlib.pyplot as plt
from multiprocessing import Pool
import itertools
import seaborn as sns
from config.config import noise_agent_config


def heat_map_plot(n_time_steps=1, rng=default_rng(0), initial_shape=50, imbalance=False):
    noise_agent_config['initial_shape'] = initial_shape
    noise_agent_config['damping_factor'] = 0.75
    noise_agent_config['imbalance_reaction'] = imbalance
    noise_agent_config['rng'] = rng    
    noise_agent_config['market_std'] = 2
    noise_agent_config['market_mean'] = 0
    noise_agent_config['limit_std'] = 2
    noise_agent_config['limit_mean'] = 0
    noise_agent_config['cancel_std'] = 2
    noise_agent_config['cancel_mean'] = 0
    # noise_agent_config['unit_volume'] = False
    if imbalance:
        noise_agent_config['initial_shape_file'] = 'initial_shape/noise.npz'
    else:
        noise_agent_config['initial_shape_file'] = 'initial_shape/noise_flow_75.npz'
    NA = NoiseAgent(**noise_agent_config)
    LOB = LimitOrderBook(list_of_agents=[NA.agent_id], level=30, only_volumes=True)
    orders = NA.initialize(time=0)
    LOB.process_order_list(orders)
    for time in range(n_time_steps):
        order, _ = NA.generate_order(LOB, time=time)
        LOB.process_order(order)
    data, orders, market_orders = LOB.log_to_df()
    heat_map(trades=market_orders, level2=data, event_times=data.time, max_level=7, scale=150, max_volume=40)
    plt.tight_layout()
    plt.title('noise+flow, c=0.75', fontsize=14)
    plt.savefig('heat_map.pdf')
    return None

heat_map_plot(n_time_steps=1000, rng=default_rng(12), imbalance=True)




