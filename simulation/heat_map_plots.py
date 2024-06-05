import os 
import sys 
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)
from config.config import noise_agent_config
from limit_order_book.limit_order_book import LimitOrderBook
from simulation.agents import NoiseAgent
from numpy.random import default_rng
from limit_order_book.plotting import plot_average_book_shape, heat_map
import matplotlib.pyplot as plt
from multiprocessing import Pool
from config.config import noise_agent_config
from performance_of_benchmarks import Market


def heat_map_plot(seed):
    M = Market(market_env='noise', execution_agent='sl_agent', volume=10, seed=seed)
    # noise_agent_config['initial_shape'] = initial_shape
    # noise_agent_config['damping_factor'] = 0.75
    # noise_agent_config['imbalance_reaction'] = imbalance
    # noise_agent_config['rng'] = rng    
    # noise_agent_config['market_std'] = 1
    # noise_agent_config['market_mean'] = 0
    # noise_agent_config['limit_std'] = 1
    # noise_agent_config['limit_mean'] = 0
    # noise_agent_config['cancel_std'] = 1
    # noise_agent_config['cancel_mean'] = 0
    # noise_agent_config['terminal_time'] = 150
    # noise_agent_config['start_time'] = 0
    # noise_agent_config['fall_back_volume'] = 5
    # noise_agent_config['unit_volume'] = False
    M.reset()
    M.run()
    # level2, orders, buy_sell = M.log_to_df()
    level2, orders, market_orders = M.lob.log_to_df()
    heat_map(trades=market_orders, level2=level2, event_times=level2.time, max_level=7, scale=250, max_volume=40)
    # plt.tight_layout()
    # plt.title('noise+flow, c=0.75', fontsize=14)
    # plt.savefig('heat_map.pdf')
    return None


heat_map_plot(seed=5)
plt.savefig('plots/heat_map.pdf')




