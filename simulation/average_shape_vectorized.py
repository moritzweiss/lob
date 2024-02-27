import sys
import os 
import time 

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from limit_order_book.plotting import plot_level2_order_book, heat_map, plot_prices, plot_average_book_shape
import matplotlib.pyplot as plt
# import gym.vector

from copy import deepcopy
from typing import Optional, Tuple
from gymnasium.spaces import Tuple, Discrete, Box
from tempfile import TemporaryFile
from matplotlib import cm 
from collections import deque, namedtuple, OrderedDict
from operator import neg
import pickle 
from sortedcontainers import SortedDict, SortedList
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import gymnasium as gym 
from gymnasium.utils.env_checker import check_env
from ray.rllib.utils.spaces.simplex import Simplex
from agents import NoiseAgent
from limit_order_book.plotting import heat_map
from limit_order_book.limit_order_book import LimitOrderBook, LimitOrder, MarketOrder, CancellationByPriceVolume, Cancellation

import pandas as pd
# to show more rows levels of the data frames 
pd.set_option('display.max_rows', 500)

# ToDo: Write everything into a config file 
# config = {'intensities': 
#           {'limit:' }
#           }


# class NoiseAgent(): 
#     def __init__(self, rng, level=30):
#         #####
#         # note: current config: cancel intensities are increased by a factor of 100
#         # volumes are half normal with mean 1 and sigma 3, they are clipped by 1, 20 
#         # the reasoning for the sigma is that events that eat into more levels of the book than the first two levels are unlikely 
#         ######

#         self.np_random = rng 
#         self.level = level
#         self.initial_level = level # number of levels to initialize
#         self.initial_bid = 1000
#         self.initial_ask = 1001
                
#         # intensities 
#         limit_intensities = np.array([0.2842, 0.5255, 0.2971, 0.2307, 0.0826, 0.0682, 0.0631, 0.0481, 0.0462, 0.0321, 0.0178, 0.0015, 0.0001])
#         self.limit_intensities = np.pad(limit_intensities, (0,level-len(limit_intensities)), 'constant', constant_values=(0))
#         cancel_intensities = 1e-3*np.array([0.8636, 0.4635, 0.1487, 0.1096, 0.0402, 0.0341, 0.0311, 0.0237, 0.0233, 0.0178, 0.0127, 0.0012, 0.0001])
#         # cancel_intensities = 5e-3*np.array([0.8636, 0.4635, 0.1487, 0.1096, 0.0402, 0.0341, 0.0311, 0.0237, 0.0233, 0.0178, 0.0127, 0.0012, 0.0001])
#         # cancel_intensities = 5e-2*np.array([0.8636, 0.4635, 0.1487, 0.1096, 0.0402, 0.034*1, 0.0311, 0.0237, 0.0233, 0.0178, 0.0127, 0.0012, 0.0001])
#         cancel_intensities = 1e-1*np.array([0.8636, 0.4635, 0.1487, 0.1096, 0.0402, 0.0341, 0.0311, 0.0237, 0.0233, 0.0178, 0.0127, 0.0012, 0.0001]) #current config !!!!!
#         # cancel_intensities = 5e-1*np.array([0.8636, 0.4635, 0.1487, 0.1096, 0.0402, 0.0341, 0.0311, 0.0237, 0.0233, 0.0178, 0.0127, 0.0012, 0.0001])
#         # cancel_intensities = 5e-1*np.array([0.8636, 0.4635, 0.1487, 0.1096, 0.0402, 0.0341, 0.0311, 0.0237, 0.0233, 0.0178, 0.0127, 0.0012, 0.0001])
#         self.cancel_intensities = np.pad(cancel_intensities, (0,level-len(cancel_intensities)), 'constant', constant_values=(0))
#         self.market_intesity = 0.1237


#         # self.market_intesity = 10*0.1237
#         # self.market_intesity = 0 

#         # volume intensities 
#         # self.market_volume_parameters = {'mean':4.0, 'sigma': 1.19} 
#         # self.limit_volume_parameters = {'mean':4.47, 'sigma': 0.83}
#         # self.cancel_volume_parameters = {'mean':4.48, 'sigma': 0.82}

#         # self.market_volume_parameters = {'mean':4.0, 'sigma': 1.19} 
#         # self.limit_volume_parameters = {'mean':4.47, 'sigma': 0.83}
#         # self.cancel_volume_parameters = {'mean':4.48, 'sigma': 0.82}

#         # half normal 
#         # stick with market: (0,3), limit: (0,3), cancel: (0,3)
#         self.market_volume_parameters = {'mean':0.00, 'sigma': 3.0} 
#         self.limit_volume_parameters = {'mean':0.00, 'sigma': 3.0}
#         self.cancel_volume_parameters = {'mean':0.00, 'sigma': 3.0}

#         # initial shape 
#         # shape = np.load('/Users/weim/projects/lob/data/stationary_shape.npz')
#         # self.initial_shape = np.mean([shape['bid'], shape['ask']], axis=0)

#         # shape = np.load('data.npz')
#         shape = np.load('data_small_queue.npz')
#         self.initial_shape = np.clip(np.rint(np.mean([shape['bid_shape'], shape['ask_shape']], axis=0)), 1, np.inf)      
#         # self.initial_shape = self.initial_level*[50]
#         self.agent_id = 'noise_agent'

#         return None  
    
#     def initialize(self): 
#         # ToDo: initial bid and ask as variable 
#         orders = [] 
#         for idx, price in enumerate(np.arange(self.initial_bid, self.initial_bid-self.initial_level, -1)):
#             order = LimitOrder(agent_id=self.agent_id, side='bid', price=price, volume=self.initial_shape[idx])
#             orders.append(order)
#         for idx, price in enumerate(np.arange(self.initial_ask, self.initial_ask+self.initial_level, 1)): 
#             order = LimitOrder(agent_id=self.agent_id, side='ask', price=price, volume=self.initial_shape[idx])
#             orders.append(order)
#         return orders


#     def sample_order(self, best_bid_price, best_ask_price, bid_volumes, ask_volumes):
#         ''''
#         - input: shape of the limit order book 
#         - output: order 
#         '''

#         # handling of nan best bid price 
#         if np.isnan(best_bid_price):
#             if np.isnan(best_ask_price):
#                 order = LimitOrder(agent_id=self.agent_id, side='bid', price=self.initial_bid, volume=self.initial_shape[0])
#             else:
#                 order = LimitOrder(agent_id=self.agent_id, side='bid', price=best_ask_price-1, volume=self.initial_shape[0])
#             return order
#         elif np.isnan(best_ask_price):
#             order = LimitOrder(agent_id=self.agent_id, side='ask', price=best_bid_price+1, volume=self.initial_shape[0])
#             return order


#         ask_cancel_intensity = np.sum(self.cancel_intensities*ask_volumes)
#         bid_cancel_intensity = np.sum(self.cancel_intensities*bid_volumes)
#         limit_intensity = np.sum(self.limit_intensities)

#         probability = np.array([self.market_intesity, self.market_intesity, limit_intensity, limit_intensity, bid_cancel_intensity, ask_cancel_intensity])
#         probability = probability/np.sum(probability)

#         action, side = self.np_random.choice([('market', 'bid'), ('market', 'ask'), ('limit', 'bid'), ('limit', 'ask'), ('cancellation', 'bid'), ('cancellation', 'ask')], p=probability)



#         # if action == 'limit':
#         #     volume = self.np_random.lognormal(self.limit_volume_parameters['mean'], self.limit_volume_parameters['sigma'])   
#         # elif action == 'market':
#         #     volume = self.np_random.lognormal(self.market_volume_parameters['mean'], self.market_volume_parameters['sigma'])
#         # elif action == 'cancellation':
#         #     volume = self.np_random.lognormal(self.cancel_volume_parameters['mean'], self.cancel_volume_parameters['sigma'])
#         # volume = np.rint(np.clip(1+np.abs(volume), 1, 2000))


#         if action == 'limit':
#             volume = self.np_random.normal(self.limit_volume_parameters['mean'], self.limit_volume_parameters['sigma'])   
#         elif action == 'market':
#             volume = self.np_random.normal(self.market_volume_parameters['mean'], self.market_volume_parameters['sigma'])
#         elif action == 'cancellation':
#             volume = self.np_random.normal(self.cancel_volume_parameters['mean'], self.cancel_volume_parameters['sigma'])
#         volume = np.rint(np.clip(1+np.abs(volume), 1, 20))

#         # volume = 1

#         if action == 'limit': 
#             probability = self.limit_intensities/np.sum(self.limit_intensities)
#             level = self.np_random.choice(np.arange(1, self.level+1), p=probability)       
#             if side == 'bid': 
#                 price = best_ask_price - level
#             else: 
#                 price = best_bid_price + level
#             order = LimitOrder(agent_id=self.agent_id, side=side, price=price, volume=volume) 

#         elif action == 'market':
#             order = MarketOrder(agent_id=self.agent_id, side=side, volume=volume)
        
#         elif action == 'cancellation':
#             if side == 'bid':
#                 probability = self.cancel_intensities*bid_volumes/np.sum(self.cancel_intensities*bid_volumes)
#                 level = self.np_random.choice(np.arange(1, self.level+1), p=probability)       
#                 price = best_ask_price - level
#             elif side == 'ask':
#                 probability = self.cancel_intensities*ask_volumes/np.sum(self.cancel_intensities*ask_volumes)
#                 level = self.np_random.choice(np.arange(1, self.level+1), p=probability)       
#                 price = best_bid_price + level
#             order = CancellationByPriceVolume(agent_id=self.agent_id, side=side, price=price, volume=volume)
            
#         return order 

#     def cancel_far_out_orders(self, lob):        
#         order_list = []
#         for price in lob.price_map['bid'].keys():
#             if price < lob.get_best_price('ask') - self.level:
#                 for order_id in lob.price_map['bid'][price]:
#                     if lob.order_map[order_id].agent_id == self.agent_id:
#                         order = Cancellation(agent_id=self.agent_id, order_id=order_id)
#                         order_list.append(order)
#         # try to establish boundary conditions 
#         # order = LimitOrder(agent_id=self.agent_id, side='bid', price=lob.get_best_price('ask') - self.level - 1, volume=self.initial_shape[0])
#         # order_list.append(order)

#         for price in lob.price_map['ask'].keys():
#             if price > lob.get_best_price('bid') + self.level:
#                 for order_id in lob.price_map['ask'][price]:
#                     if lob.order_map[order_id].agent_id == self.agent_id:
#                         order = Cancellation(agent_id=self.agent_id, order_id=order_id)
#                         order_list.append(order)                                
#         # order = LimitOrder(agent_id=self.agent_id, side='ask', price=lob.get_best_price('bid') + self.level + 1, volume=self.initial_shape[0])
#         # order_list.append(order)

#         return order_list


# # simulation only, no benchmark strategies 

class Market(gym.Env):    
    def __init__(self, seed=0, terminal_time=int(1e3), level=30):
        self.level = 30 
        self.terminal_time = terminal_time
        super().reset(seed=seed)
        self.noise_agent = NoiseAgent(level=30, rng=self.np_random, initial_shape_file='data_small_queue.npz', config_n=1, imbalance_reaction=True, imbalance_n_levels=4)
        self.observation_space = Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.action_space = Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        return None

    def reset(self) :
        self.lob = LimitOrderBook(list_of_agents=[self.noise_agent.agent_id], level=self.level)
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
        # bid_prices, bid_volumes = self.lob.data.bid_prices[-1], self.lob.data.bid_volumes[-1]
        # ask_prices, ask_volumes = self.lob.data.ask_prices[-1], self.lob.data.ask_volumes[-1]
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
        plt.show()


    else:

        level = 30
        T = 1e3

        env = Market(seed=int(2), terminal_time=int(T))
        
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

        print(f'time elapsed in minutes: {(time.time()-start)/60}')

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

        # bid_volumes = np.vstack(env.lob.data.bid_volumes[-int(T/2):][::reduction])
        # ask_volumes = np.vstack(env.lob.data.ask_volumes[-int(T/2):][::reduction])
        # bid_shape = np.mean(bid_volumes[-int(T/2):][::reduction] , axis=0)
        # ask_shape = np.mean(ask_volumes[-int(T/2):][::reduction], axis=0)    

        df, trades = env.lob.log_to_df()

        plot_average_book_shape(bid_volumes=env.lob.data.bid_volumes, ask_volumes=env.lob.data.ask_volumes, level=30, symetric=False)

        plot_prices(level2=df, trades=trades, marker_size=100)

        heat_map(trades=trades, level2=df, max_level=5, scale=500, max_volume=50)

        # reduction = int(T/100)
        # bid_volumes = np.vstack(env.lob.data.bid_volumes[-int(T/2):][::reduction])
        # ask_volumes = np.vstack(env.lob.data.ask_volumes[-int(T/2):][::reduction])
        # bid_shape = np.mean(bid_volumes[-int(T/2):][::reduction] , axis=0)
        # ask_shape = np.mean(ask_volumes[-int(T/2):][::reduction], axis=0)    

        # BE CAREFULE ABOUT SAVING THIS FILE
        # np.savez('data_small_queue_.npz', bid_shape=bid_shape, ask_shape=ask_shape)    
                
        plt.show()



