from typing import Any
from agents import NoiseAgent
import gymnasium as gym 
from limit_order_book.limit_order_book import LimitOrderBook, MarketOrder, LimitOrder
from limit_order_book.plotting import heat_map
import matplotlib.pyplot as plt
import numpy as np
from limit_order_book.plotting import plot_average_book_shape

# script to generate average book shape, histogram of traded volume, spreads 

class Market(gym.Env):
    def __init__(self, seed,  terminal_time=int(1e3), level=30, imbalance_reaction=False, damping_factor=0.5) -> None:
        self.terminal_time = terminal_time
        self.noise_agent = NoiseAgent(level=level, rng=np.random.default_rng(seed) , imbalance_reaction=imbalance_reaction, initial_shape_file='data_small_queue.npz', config_n=1, damping_factor=damping_factor)
        # summary statistics
        self.spreads = []
        self.bid_volumes = []
        self.ask_volumes = []
        self.traded_volumes = []
        return None 
        
    def reset(self):
        self.lob = LimitOrderBook(level=self.noise_agent.level, list_of_agents=[self.noise_agent.agent_id])
        orders = self.noise_agent.initialize()
        [self.lob.process_order(order) for order in orders]
        self.time = 0 
        return None 
    
    def step(self):
        if self.time == self.terminal_time:
            return True, self.final_info()
        order = self.noise_agent.sample_order(self.lob.data.best_bid_prices[-1], self.lob.data.best_ask_prices[-1], self.lob.data.bid_volumes[-1], self.lob.data.ask_volumes[-1])
        self.lob.process_order(order)
        self.time += 1
        return False, {}
    
    def final_info(self):
        # bid ask shapes
        bid_volumes = self.lob.data.bid_volumes[100::20]
        ask_volumes = self.lob.data.ask_volumes[100::20]
        # bid_shape = np.mean(bid_volumes, axis=0)
        # ask_shape = np.mean(ask_volumes, axis=0)
        # spreads
        ask_prices = self.lob.data.best_ask_prices[100::20]
        bid_prices = self.lob.data.best_bid_prices[100::20]
        spreads = [ask - bid for ask, bid in zip(ask_prices, bid_prices)]
        # spread = np.mean(ask_prices - bid_prices)
        # average traded volume 
        m_order_size = [order.volume for order in self.lob.data.orders if isinstance(order, MarketOrder)]
        m_order_size = sum(m_order_size)
        self.spreads.extend(spreads)
        self.bid_volumes.extend(bid_volumes)
        self.ask_volumes.extend(ask_volumes)
        self.traded_volumes.append(m_order_size)
        return {'spreads': spreads, 'bid_volumes': bid_volumes, 'ask_shape': ask_volumes, 'traded_volume': m_order_size}
    
    def sample_episodes(self, n_episodes):
        for _ in range(n_episodes):
            self.reset()
            terminated = False 
            while not terminated:
                terminated, info = self.step()
            # print(info)        
        return {'spreads': np.mean(self.spreads), 'bid_volumes': np.mean(self.bid_volumes, axis=0), 'ask_volumes': np.mean(self.ask_volumes, axis=0), 'traded_volumes': np.mean(self.traded_volumes)}


if __name__ == '__main__':
    # ToDO: implement benchmarks market, linear submit and leave 
    M = Market(seed=2, imbalance_reaction=False,  terminal_time=1000, level=30, damping_factor=1.0)
    info = M.sample_episodes(500)
    print(info)

    # plt.hist(M.spreads)

    plt.figure()
    plot_average_book_shape(M.bid_volumes, M.ask_volumes, level=30)
    plt.savefig('average_book_shape.pdf')
    
    # spreads
    plt.figure()
    plt.hist(M.spreads, bins=np.arange(1, 6), color='green')
    plt.title('spreads')
    plt.fontsize = 16
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(1, 5)
    plt.savefig('spreads.pdf') 

    # traded volumes 
    plt.figure()
    plt.hist(M.traded_volumes, bins=35, color='green')
    plt.title('traded volume')
    plt.fontsize = 16
    plt.xticks(fontsize=16)
    plt.savefig('traded_volume.pdf')

    
    # heat_map(trades=orders, level2=data, max_level=5, max_volume=30, scale=500)
    # plt.show()