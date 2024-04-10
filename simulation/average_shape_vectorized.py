import sys
import os 
import time 
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from limit_order_book.plotting import plot_level2_order_book, heat_map, plot_prices, plot_average_book_shape
import matplotlib.pyplot as plt
from gymnasium.spaces import Tuple, Discrete, Box
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym 
from agents import NoiseAgent
from limit_order_book.plotting import heat_map
from limit_order_book.limit_order_book import LimitOrderBook, LimitOrder, MarketOrder, CancellationByPriceVolume, Cancellation
import pandas as pd
pd.set_option('display.max_rows', 500)


class Market(gym.Env):    
    def __init__(self, seed=0, terminal_time=int(1e3), level=30):
        self.level = 30 
        self.terminal_time = terminal_time
        super().reset(seed=seed)
        self.noise_agent = NoiseAgent(level=30, rng=self.np_random, initial_shape_file='data_small_queue.npz', config_n=1, imbalance_reaction=True, imbalance_n_levels=4, damping_factor=0.5)
        self.observation_space = Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.action_space = Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        return None

    def reset(self) :
        self.lob = LimitOrderBook(list_of_agents=[self.noise_agent.agent_id], level=self.level, only_volumes=False)
        self.time = 0 
        orders = self.noise_agent.initialize()
        [self.lob.process_order(order) for order in orders]
        # initialize the order book with some orders         
        return None, {}
    
    def step(self, action=np.array([0.0])):
        # process order 
        truncated = terminated = False
        # order = sample order 
        order = self.noise_agent.sample_order(self.lob.data.best_bid_prices[-1], self.lob.data.best_ask_prices[-1], self.lob.data.bid_volumes[-1], self.lob.data.ask_volumes[-1])
        # note: logging or note makes a big time difference 
        self.lob.process_order(order)      
        orders = self.noise_agent.cancel_far_out_orders(lob=self.lob)
        [self.lob.process_order(order) for order in orders]  
        self.time += 1
        if self.time >= self.terminal_time:
            terminated = True
            truncated = True
            print('terminal time reached')
            bid_prices = self.lob.data.bid_prices[-int(self.terminal_time/2):][::100]
            bid_volumes = self.lob.data.bid_volumes[-int(self.terminal_time/2):][::100]
            ask_prices = self.lob.data.ask_prices[-int(self.terminal_time/2):][::100]
            ask_volumes = self.lob.data.ask_volumes[-int(self.terminal_time/2):][::100]
            return None, None, terminated, truncated, {'bid_prices':bid_prices, 'bid_volumes':bid_volumes, 'ask_prices':ask_prices, 'ask_volumes':ask_volumes, 'time':self.lob.update_n}
        return None, None, terminated, truncated, {}


def concat_lists(l):
    x = []
    [x.extend(item) for item in l]
    return x

if __name__ == '__main__':
    # note: the set up with n_envs=8, level=30, T=1e6, initial_shape=50, gives relatively stable results: take average shape of bid and ask side as a starting point. it runs about 24 minutes. 
    parallel = False

    if parallel:
        level = 30
        n_envs = 8
        T = 1e5

        env_functions = [lambda: Market(seed=int(100*seed), terminal_time=int(T)) for seed in range(n_envs)]
        
        env = gym.vector.AsyncVectorEnv(env_functions)
        
        start = time.time()
        
        env.reset()    

        terminated = False 
        t = 1 
        while not terminated:
            action = env.action_space.sample()
            out = env.step(action)
            terminated = out[2][0]
            if t%int(1e4) == 0:
                print(t)
            t += 1 

        print(f'time elapsed in minutes: {(time.time()-start)/60}')

        bid_volumes = concat_lists([out[4]['final_info'][idx]['bid_volumes'] for idx in range(n_envs)])
        bid_volumes = np.vstack(bid_volumes)
        ask_volumes = concat_lists([out[4]['final_info'][idx]['ask_volumes'] for idx in range(n_envs)])
        ask_volumes = np.vstack(ask_volumes)

        print(f'time elapsed after stacking in minutes: {(time.time()-start)/60}')

        # bid_prices = np.stack([out[4]['final_info'][idx]['bid_prices'][-int(T/2):] for idx in range(n_envs)])
        # bid_volumes = np.stack([out[4]['final_info'][idx]['bid_volumes'][-int(T/2):] for idx in range(n_envs)])
        # ask_prices = np.stack([out[4]['final_info'][idx]['ask_prices'][-int(T/2):] for idx in range(n_envs)])
        # ask_volumes = np.stack([out[4]['final_info'][idx]['ask_volumes'][-int(T/2):] for idx in range(n_envs)])

        # data, orders = env.lob.log_to_df()
        # bid_volumes = np.reshape(bid_volumes,  (-1,level))
        # ask_volumes = np.reshape(ask_volumes,  (-1,level))
        bid_shape = np.mean(bid_volumes, axis=0)
        ask_shape = np.mean(ask_volumes, axis=0)    

        # np.savez('data_new_param.npz', bid_shape=bid_shape, ask_shape=ask_shape)    
        plt.figure()
        plt.bar(np.arange(1000,1000-level,-1), bid_shape, color='b')
        plt.bar(np.arange(1001,1001+level,1), ask_shape, color='r')
        plt.savefig('average_shape.pdf')


    else:

        level = 30
        T = 1e4

        env = Market(seed=int(1), terminal_time=int(T))
        
        start = time.time()
        
        env.reset()    

        terminated = False 
        t = 1 
        while not terminated:
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            if t%int(1e4) == 0:
                print(t)
            t += 1 

        print(f'time elapsed in seconds: {(time.time()-start)}')

        # shapes 
        reduction = int(T/100)
        bid_volumes = np.vstack(env.lob.data.bid_volumes[::reduction])
        ask_volumes = np.vstack(env.lob.data.ask_volumes[::reduction])
        bid_shape = np.mean(bid_volumes, axis=0)
        ask_shape = np.mean(ask_volumes, axis=0)    
        print(f'time elapsed after stacking in minutes: {(time.time()-start)/60}')
        print('ask shape')
        print(ask_shape- env.noise_agent.initial_shape)
        print('bid shape - saved initial shape')
        print(bid_shape - env.noise_agent.initial_shape)

        # plt.figure()
        # plt.bar(np.arange(1000,1000-level,-1), bid_shape - env.noise_agent.initial_shape, color='b')
        # plt.bar(np.arange(1001,1001+level,1), ask_shape - env.noise_agent.initial_shape, color='r')
        # plt.savefig('average_shape_difference.pdf')
        # plt.show()

        bid_volumes = np.vstack(env.lob.data.bid_volumes[-int(T/2):][::reduction])
        ask_volumes = np.vstack(env.lob.data.ask_volumes[-int(T/2):][::reduction])
        bid_shape = np.mean(bid_volumes[-int(T/2):][::reduction] , axis=0)
        ask_shape = np.mean(ask_volumes[-int(T/2):][::reduction], axis=0)    

        df, trades = env.lob.log_to_df()

        plot_average_book_shape(bid_volumes=env.lob.data.bid_volumes, ask_volumes=env.lob.data.ask_volumes, level=30, symetric=True)
        plt.tight_layout()
        plt.savefig('average_shape.pdf')

        # plot_prices(level2=df, trades=trades, marker_size=100)

        # heat_map(trades=trades, level2=df, max_level=5, scale=500, max_volume=50)

        plt.show()

        # reduction = int(T/100)
        # bid_volumes = np.vstack(env.lob.data.bid_volumes[-int(T/2):][::reduction])
        # ask_volumes = np.vstack(env.lob.data.ask_volumes[-int(T/2):][::reduction])
        # bid_shape = np.mean(bid_volumes[-int(T/2):][::reduction] , axis=0)
        # ask_shape = np.mean(ask_volumes[-int(T/2):][::reduction], axis=0)    

        # BE CAREFULE ABOUT SAVING THIS FILE
        # np.savez('data_small_queue_.npz', bid_shape=bid_shape, ask_shape=ask_shape)    
                
        # plt.savefig('average_shape.pdf')



