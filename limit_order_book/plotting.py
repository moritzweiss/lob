# Get the current script's directory
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)# Add the parent directory to sys.path
sys.path.append(parent_dir)
# import 
import numpy as np
# from advanced_multi_lot import Market 
import matplotlib.pyplot as plt
from matplotlib import cm 
import pandas as pd 

# TODO: Wrap all the data into one data frame 


def heat_map(trades, level2, max_level=30, scale=1000, max_volume=1000):
    '''
    inputs:
        - trades: data frame with columns ['type', 'side', 'size', 'price']
        - level2: data frame with columns ['best_bid_price', 'best_ask_price', 'best_bid_volume', 'best_ask_volume', 'bid_price_0', 'bid_volume_0', 'ask_price_0', 'ask_volume_0', ...]

    output:
        - out: heatmap of the book 
    '''

    bid_prices = [f'bid_price_{n}' for n in range(max_level)] 
    bid_prices= np.hstack(np.array(level2[bid_prices]))
    ask_prices = [f'ask_price_{n}' for n in range(max_level)]
    ask_prices = np.hstack(np.array(level2[ask_prices]))
    bid_volumes = [f'bid_volume_{n}' for n in range(max_level)]
    bid_volumes = -1*np.hstack(np.array(level2[bid_volumes]))
    ask_volumes = [f'ask_volume_{n}' for n in range(max_level)]
    ask_volumes = np.hstack(np.array(level2[ask_volumes]))

    time = np.array(level2.index)
    N = len(time)

    prices = np.hstack([bid_prices, ask_prices])
    volumes = np.hstack([bid_volumes, ask_volumes])
    time = np.arange(N)
    extended_time = []
    for n in range(N):
        extended_time.extend(max_level*[time[n]])
    for n in range(N):
        extended_time.extend(max_level*[time[n]])

    trades = trades.shift(-1)
    bid_mask = (trades.side == 'bid') & (trades.type == 'M')
    ask_mask = (trades.side == 'ask') & (trades.type == 'M')     
    # max_volume = max(trades['size'][trades.type == 'M'])
    # hard coded. find better logic for this. 
    # max_volume = 1000

    plt.figure()

    plt.scatter(extended_time, prices, c=volumes, cmap=cm.seismic, vmin=-max_volume, vmax=max_volume)

    plt.plot(time, level2.best_bid_price, '-', color='black', linewidth=3)
    plt.plot(time, level2.best_ask_price, '-', color='black', linewidth=3)
    plt.colorbar()

    M = max(trades['size'])
    plt.scatter(time[ask_mask], level2.best_ask_price[ask_mask], color='black', marker='^', s= (scale/M)*trades['size'][ask_mask]) 
    plt.scatter(time[bid_mask], level2.best_bid_price[bid_mask], color='black', marker='v', s= (scale/M)*trades['size'][bid_mask])


    return None  

def plot_average_book_shape(bid_volumes, ask_volumes, level=3, symetric=False):
    """
    - bid/ask_volumes: list of np arrays, [v1, v2, v3, ...]
    """ 
    level = len(bid_volumes[0]) 
    book_shape_bid = np.nanmean(bid_volumes, axis=0)
    book_shape_ask = np.nanmean(ask_volumes, axis=0)    
    if symetric:
        plt.figure()        
        shape = (book_shape_bid + book_shape_ask)/2
        plt.bar(range(0,-level,-1), shape, color='blue', label='bid')
        plt.bar(range(1,level+1,1), shape, color='red', label='ask')    
    else:
        plt.figure()        
        plt.bar(range(0,-level,-1), book_shape_bid, color='blue', label='bid')
        plt.bar(range(1,level+1,1), book_shape_ask, color='red', label='ask')
    plt.legend(loc='upper right')
    plt.xlabel('relative distance to mid price')
    plt.ylabel('average volume')

    # TODO: analyze average book shape
    return None 

def plot_prices(level2, trades, marker_size=50):
    """
    the method plots 
        - bid and ask prices 
        - microprice 
        - trades on bid and ask (larger trade with larger marker size)            
    """

    level2 = level2.copy()

    level2['micro_price'] = (level2.best_bid_price * level2.best_ask_volume + level2.best_ask_price * level2.best_bid_volume) / (level2.best_bid_volume + level2.best_ask_volume)

    trades = trades.shift(-1)

    data = pd.concat([level2, trades], axis=1)

    bid_mask = (data.type == 'M') & (data.side == 'bid')
    ask_mask = (data.type == 'M') & (data.side == 'ask')
    max_volume = max(data['size'][data.type == 'M'])

    plt.figure()

    plt.plot(data.index, data.best_bid_price, '--', color='grey')
    plt.plot(data.index, data.best_ask_price, '--', color='grey')
    plt.plot(data.index, data.micro_price , '-', color='blue')


    plt.scatter(data.index[bid_mask], data.best_bid_price[bid_mask], color='red', marker='v', s= marker_size*data['size'][bid_mask]/max_volume)
    plt.scatter(data.index[ask_mask], data.best_ask_price[ask_mask], color='green', marker='^',s= marker_size*data['size'][ask_mask]/max_volume)

    
    return None 



def plot_level2_order_book(bid_prices, ask_prices, bid_volumes, ask_volumes, n):
    """"
    input: 
        - bid/ask_prices: list of np arrays, [p1, p2, p3, ...]
        - bid/ask_volumes: list of np arrays, [v1, v2, v3, ...]
        - n: index of the order book snapshot
    output:
        - plot of the order book snapshot
    """
    plt.figure()
    plt.bar(bid_prices[n], bid_volumes[n], color='b')
    plt.bar(ask_prices[n], ask_volumes[n], color='r')
    return


if __name__ == '__main__': 
    config = {'total_n_steps': int(1e3), 'log': True, 'seed':0, 'initial_volume': 500, 'env_type': 'simple', 'ada':False}
    M = Market(config=config)
    print(f'initial volume is {config["initial_volume"]}')
    rewards = []
    for n in range(1):
        observation, _ = M.reset()
        assert observation in M.observation_space 
        terminated = truncated = False 
        reward_per_episode = 0 
        while not terminated and not truncated: 
            action = np.array([0, 1, 0, 0, 0], dtype=np.float32)
            assert action in M.action_space
            observation, reward, terminated, truncated, info = M.step(action, transform_action=False)
            assert observation in M.observation_space
            reward_per_episode += reward
        rewards.append(reward_per_episode)
        assert M.volume == 0 
    
    # ToDo: either make history all list or all np arrays   
    # Check logging mechanism in LOB 
    # plot_level2_order_book(M.bid_prices, M.ask_prices, M.bid_volumes, M.ask_volumes, 0)
    # plot_average_book_shape(M.bid_volumes, M.ask_volumes)
    # plot_prices(M.best_bid_prices, M.best_ask_prices, M.best_bid_volumes, M.best_ask_volumes, M.trades)
    plt.show()
    # plt.figure()
    heat_map(np.array(M.best_ask_prices), np.array(M.best_bid_prices), M.bid_prices, M.ask_prices, M.bid_volumes, M.ask_volumes, M.trades)
    plt.show()