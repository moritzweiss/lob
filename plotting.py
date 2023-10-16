import numpy as np
from advanced_multi_lot import M
import matplotlib.pyplot as plt
from matplotlib import cm 

# TODO: can rewrite as a class which takes LOB class or LOB.data as input 
# make logging in the LOB more standardized 


def heat_map(best_ask_prices, best_bid_prices, bid_prices, ask_prices, bid_volumes, ask_volumes, trades, max_level=30, y_lim = (995, 1005) ):
    '''
    - bid/ask_prices: list of integer values np arrays. each array contains [p1, p2, p3, ...] 
    - bid/ask_volumes: list of integer values np arrays. each array contains [v1, v2, v3, ...]
    - best_bid/ask_prices: list of integer values
    - trades: list of market trades: None or ('bid', size) or ('ask', size), None means that there was no trade
    - out: heatmap of the book 
    '''

    # color settings / this is code to create a custom colormap
    # N = 128
    # lightness = 108
    # reds = np.ones((N, 4))
    # reds[:, 0] = np.linspace(1, 1, N)
    # reds[:, 1] = np.linspace(lightness / N, 0, N)
    # reds[:, 2] = np.linspace(lightness / N, 0, N)
    # blues = np.ones((N, 4))
    # blues[:, 0] = np.linspace(0, lightness / N, N)
    # blues[:, 1] = np.linspace(0, lightness / N, N)
    # blues[:, 2] = np.linspace(1, 1, N)
    # newcolors = np.vstack([blues, reds])

    # scatter plot of bid ask prices and volumes 
    prices = np.hstack((np.hstack(bid_prices),np.hstack(ask_prices)))
    volumes = np.hstack((-1*np.hstack(bid_volumes),np.hstack(ask_volumes)))
    N = len(bid_prices)
    time = np.arange(N)
    extended_time = []
    for n in range(N):
        extended_time.extend(max_level*[time[n]])
    for n in range(N):
        extended_time.extend(max_level*[time[n]])
    plt.scatter(extended_time, prices, c=volumes, cmap=cm.seismic, vmin=-2000, vmax=2000)

    # plot best bid and ask prices
    plt.plot(time, best_ask_prices, '-', color='black', linewidth=3)
    plt.plot(time, best_bid_prices, '-', color='black', linewidth=3)
    plt.colorbar()

    # set x and y limits 
    plt.ylim(bottom=y_lim[0], top=y_lim[1])

    # plot trades 
    market_ask = []    
    volumes = []    
    for x in trades:
        if x is None:
            market_ask.append(False)
            volumes.append(0)
        elif x[0] == 'bid':
            market_ask.append(False)
            volumes.append(x[1])
        else:
            market_ask.append(True)
            volumes.append(x[1])

    market_bid = []
    for x in trades:
        if x is None:
            market_bid.append(False)
        elif x[0] == 'ask':
            market_bid.append(False)
        else:
            market_bid.append(True)

    volumes = np.array(volumes)
    plt.scatter(time[market_ask], best_ask_prices[market_ask], color='black', marker='^', s=0.5*volumes[market_ask]) 
    plt.scatter(time[market_bid], best_bid_prices[market_bid], color='black', marker='v', s=0.5*volumes[market_bid])

    return None  

def plot_average_book_shape(bid_volumes, ask_volumes, level=30):
    """
    - bid/ask_volumes: list of np arrays, [v1, v2, v3, ...]
    """ 
    book_shape_bid = np.mean(bid_volumes, axis=0)
    book_shape_ask = np.mean(ask_volumes, axis=0)    
    plt.figure()
    plt.bar(range(0,-level,-1), book_shape_bid, color='red', label='bid')
    plt.bar(range(1,level+1,1), book_shape_ask, color='blue', label='ask')
    plt.legend(loc='upper right')
    plt.xlabel('relative distance to mid price')
    plt.ylabel('average volume')

    # TODO: analyze average book shape
    return None 

def plot_prices(best_bid_prices, best_ask_prices, best_bid_volumes, best_ask_volumes, trades, level=30):
    """
    the method plots 
        - bid and ask prices 
        - microprice 
        - trades on bid and ask (larger trade with larger marker size)            
    """

    bid_prices = np.array(best_bid_prices)
    ask_prices = np.array(best_ask_prices)
    bid_volume = np.array(best_bid_volumes)
    ask_volume = np.array(best_ask_volumes)
    microprice = (bid_prices * ask_volume + ask_prices * bid_volume) / (bid_volume + ask_volume)
    time = np.arange(0, len(bid_prices))


    trades = [x[0] if x is not None else False for x in trades]
    bid_mask = [True if x == 'bid' else False for x in trades]
    ask_mask = [True if x == 'ask' else False for x in trades]

    plt.figure()
    plt.plot(time, bid_prices, '--', color='grey')
    plt.plot(time, ask_prices, '--', color='grey')
    plt.plot(time, microprice, '-', color='blue')

    plt.scatter(time[bid_mask], bid_prices[bid_mask], color='red', marker='x')
    plt.scatter(time[ask_mask], ask_prices[ask_mask], color='green', marker='x')

    # TODO: add sizes and triangles as above for market orders r
    
    return None 



def plot_level2_order_book(bid_prices, ask_prices, bid_volumes, ask_volumes, n):
    """"
    - bid/ask_prices: list of np arrays, [p1, p2, p3, ...]
    - bid/ask_volumes: list of np arrays, [v1, v2, v3, ...]
    """
    plt.figure()
    plt.bar(bid_prices[n], bid_volumes[n], color='b')
    plt.bar(ask_prices[n], ask_volumes[n], color='r')
    return


if __name__ == '__main__': 
    config = {'total_n_steps': int(1e3), 'log': True, 'seed':0, 'initial_volume': 500, 'env_type': 'simple', 'ada':False}
    M = M(config=config)
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
    plot_prices(M.best_bid_prices, M.best_ask_prices, M.best_bid_volumes, M.best_ask_volumes, M.trades)
    plt.show()
    # plt.figure()
    # heat_map(np.array(M.best_ask_prices), np.array(M.best_bid_prices), M.bid_prices, M.ask_prices, M.bid_volumes, M.ask_volumes, M.trades)
    # plt.show()