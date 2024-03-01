# from typing import List, Optional, Tuple, Union
# from average_shape_vectorized import NoiseAgent
# from limit_order_book.limit_order_book import Li÷mitOrder
# from limit_order_book.plotting import heat_map

import matplotlib.pyplot as plt
import numpy as np
from agents import NoiseAgent
import gymnasium as gym 
from limit_order_book.limit_order_book import LimitOrderBook
from limit_order_book.plotting import heat_map


class Market(gym.Env):
    def __init__(self, seed, n_steps=int(1e3), level=30) -> None:
        super().reset(seed=seed)
        self.level = level
        self.n_steps = n_steps
        self.noise_agent = NoiseAgent(level=level, rng=self.np_random, config_n=1, initial_shape_file='data_small_queue.npz', imbalance_reaction=False)
        return None 
    
    def reset(self):
        self.lob = LimitOrderBook(level=self.level, list_of_agents=['smart_agent','noise_agent'])
        self.time = 0
        orders = self.noise_agent.initialize()
        [self.lob.process_order(order) for order in orders]
        return None
    
    def step(self):
        terminated = False
        order = self.noise_agent.sample_order(self.lob.data.best_bid_prices[-1], self.lob.data.best_ask_prices[-1], self.lob.data.bid_volumes[-1], self.lob.data.ask_volumes[-1])
        out = self.lob.process_order(order)
        if self.time == self.n_steps:
            terminated = True
        self.time += 1
        return terminated


n_samples = int(5e2)
max_steps = int(1e3)
volume = 100 
M = Market(seed=1, n_steps=max_steps)


mid_prices = []
market_order_volumes = []
spreads = []
for n_samples in range(n_samples):
    terminated = False
    M.reset()
    if n_samples % 100 == 0:
        print(n_samples)
    while not terminated:
        terminated = M.step()
        spreads.append(M.lob.data.best_ask_prices[-1] - M.lob.data.best_bid_prices[-1])
        if terminated:
            mid_prices.append((M.lob.data.best_bid_prices[-1] + M.lob.data.best_ask_prices[-1])/2)
            m = sum([order.volume if order.type == 'market' else 0 for order in M.lob.data.orders])
            market_order_volumes.append(m)
            break

# heat map 
# df, trades = M.lob.log_to_df()
# heat_map(trades=trades, level2=df, max_volume=100, scale=200, max_level=5)
# plt.show()

print(f'average trade volume is {np.mean(market_order_volumes)}')
print(f'average mid price is {np.mean(mid_prices)}')
print(f'average spread is {np.mean(spreads)}')


# mid price histogram

plt.hist(mid_prices, bins=np.arange(990, 1011), color='green', align='mid')
plt.title('mid prices range')
plt.fontsize = 12
plt.xticks(fontsize=12)
plt.xticks(np.arange(990, 1011, 2))
plt.savefig('plots/mid_prices_range.pdf')


# spread histogram
plt.figure()
plt.hist(spreads, bins=np.arange(1, 6), color='green')
plt.title('spreads')
plt.fontsize = 16
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlim(1, 5)
plt.savefig('plots/spreads.pdf')

# order size histogram
plt.figure()
plt.hist(market_order_volumes, bins=35, color='green')
plt.title('traded volume')
plt.fontsize = 16
plt.xticks(fontsize=16)
plt.savefig('plots/traded_volume.pdf')


# show plots 
plt.show()
