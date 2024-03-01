from collections import deque, namedtuple, OrderedDict
import pickle 
from sortedcontainers import SortedDict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# TODO:

# - implement LOB simulator based on Cont paper (first take what is already on Github). Check the volatilty of the LOB simulator.
# - take parameters as from the book (with unit size).
# - implement DQN agent.


# TODO: 
# - seperate LOB, from Simulation, from Agent, from Plotting 
# - the structure could bef


# - LOB (matching engine).
# - Simulation Agent (generating Poisson order flows).
# - Trainable Agent (using DQN).
# - Plotting functions .

# Issues:  
# - Find realistic parameters: what is a typical large tick instrument which QB trades? 
# - How to deal with low volatility: trade time, price time (only when bid or asks moves).
# - Large queu sizes mean that there is very limited volatility.



class OrderBook():
    def __init__(self):
        self.order_map = {'bid': {}, 'ask': {}}
        self.price_map = {'bid': SortedDict(), 'ask': SortedDict()}
        self.time = 0 
        self.order_id = 0
        self.mid_price_history = list()
        self.ask_price_history = list()
        self.bid_price_history = list()
        self.bid_volume_history = list()
        self.ask_volume_history = list()
        self.ask_market_order_history = list()
        self.bid_market_order_history = list()
        self.time_history = list()
        self.lob_shape_history = {'bid': list(), 'ask': list()}
        self.order_book_history = list()

    def logging(self, trade=False, order_book=True):
        if self.price_map['bid'] and self.price_map['ask']:
            # could reconstruct the order book after the simulation 
            if order_book:
                self.order_book_history.append(self.order_book_dict())
            self.mid_price_history.append(self.get_mid_price())
            self.ask_price_history.append(self.get_best_ask())
            self.bid_price_history.append(self.get_best_bid())
            self.bid_volume_history.append(self.get_best_bid_volume())
            self.ask_volume_history.append(self.get_best_ask_volume())
            self.time_history.append(self.time)
            if trade == 'ask':
                self.ask_market_order_history.append(1)
                self.bid_market_order_history.append(0)
            elif trade == 'bid':
                self.ask_market_order_history.append(0)
                self.bid_market_order_history.append(1)        
            else:
                self.ask_market_order_history.append(0)
                self.bid_market_order_history.append(0)
        else:
            return None 

    def get_mid_price(self):
        best_bid = self.price_map['bid'].peekitem(-1)[0]
        best_ask = self.price_map['ask'].peekitem(0)[0]
        return (best_bid + best_ask) / 2
    
    def get_best_bid_volume(self):
        return len(self.price_map['bid'].peekitem(-1)[1])
    
    def get_best_ask_volume(self):
        return len(self.price_map['ask'].peekitem(0)[1])

    def get_best_bid(self):
        return self.price_map['bid'].peekitem(-1)[0]
    
    def get_best_ask(self):
        return self.price_map['ask'].peekitem(0)[0]
    
    def limit_order(self, order):      
        """
        - assign order id and time to the order
        - if limit price is not in price map, create new price level
        - each price level is an ordered dict
        - if limit price is in price map, add order to the price level

        Args:
            order is a dict with keys: type, side, price. we will add id and time to the order.
        
        Returns:
            None. Changes the state of the order book internally. 
        """
        self.logging()
        assert order['type'] == 'limit'
        order['id'] = self.order_id
        order['time'] = self.time
        self.order_map[order['side']][order['id']] = {'price': order['price'], 'time': order['time']}
        if order['price'] in self.price_map[order['side']]:
            self.price_map[order['side']][order['price']][order['id']] = order['time']
        else:
            self.price_map[order['side']][order['price']] = OrderedDict({order['id']: order['time']}) 
        self.order_id += 1
        self.time += 1
        return None 
    
    def market_order(self, order):
        """
        - match order against limt order in the book
        - remove the order from the book 
        """
        self.logging(trade=order['side'])
        assert order['type'] == 'market'
        if order['side'] == 'ask':
            idx = 0
        if order['side'] == 'bid':
            idx = -1 
        price, _ = self.price_map[order['side']].peekitem(idx)
        order_id, _ = self.price_map[order['side']][price].popitem(last=False)
        self.order_map[order['side']].pop(order_id)
        self.handle_empty_price_level(side=order['side'], price=price)
        self.time += 1 
        return None
    
    def cancellation(self, order):
        """
        remove order with corresponding id from the order map
        """
        self.logging()
        assert order['type'] == 'cancel'
        price_info = self.order_map[order['side']].pop(order['id'])        
        self.price_map[order['side']][price_info['price']].pop(order['id'])
        self.handle_empty_price_level(side=order['side'], price=price_info['price'])
        self.time += 1 
        return None

    def handle_empty_price_level(self, side, price):
        """
        - remove an empty price level from the price map 
        - raise error if the side is empty
        """
        if not self.price_map[side][price]:
            self.price_map[side].pop(price)        
            if not self.price_map[side]:
                raise ValueError(f"{side} side is empty!")
        return None 

    def get_limit_order_book_shape(self):
        first_ask_levels = [self.get_best_bid() + p for p in range(1, 21, 1)]
        first_bid_levels = [self.get_best_ask() - p for p in range(1, 21, 1)]
        # bid_prices = list(self.price_map['bid'].keys())
        # ask_prices = list(self.price_map['ask'].keys())
        bid_volumes = [len(self.price_map['bid'][price]) if price in self.price_map['bid'] else 0 for price in first_bid_levels]
        ask_volumes = [len(self.price_map['ask'][price]) if price in self.price_map['ask'] else 0 for price in first_ask_levels]
        return first_bid_levels, bid_volumes, first_ask_levels, ask_volumes
    
    def log_limit_order_book_shape(self):
        _, bid_volumes, _, ask_volumes = self.get_limit_order_book_shape()
        self.lob_shape_history['bid'].append(bid_volumes)
        self.lob_shape_history['ask'].append(ask_volumes)
        return None
    
    def mid_price_plot(self):
        time = np.array(self.time_history)
        mask = time>=self.init_time
        time = time[mask]
        ask = np.array(self.ask_price_history)[mask]
        bid = np.array(self.bid_price_history)[mask]
        ask_volume = np.array(self.ask_volume_history)[mask]
        bid_volume = np.array(self.bid_volume_history)[mask]
        micro_price = (ask*bid_volume + bid*ask_volume)/(ask_volume + bid_volume)
        buy_market = np.array(self.ask_market_order_history)[mask]
        sell_market = np.array(self.bid_market_order_history)[mask]
        plt.plot(time, bid, '--', color='grey', label='bid/ask')
        plt.plot(time, ask, '--', color='grey')
        plt.plot(time, micro_price, '-', color='blue', label='micro price')
        plt.scatter(time[buy_market==1], ask[buy_market==1], color='red', marker='x')
        plt.scatter(time[sell_market==1], bid[sell_market==1], color='red', marker='x', label='market order')
        plt.xlim(left=time[0], right=time[-1])
        plt.legend(loc='best')
        plt.xlabel('tick time')
        plt.ylabel('relative price')
        return None
    
    def order_book_dict(self):
        order_book = {'bid': {}, 'ask': {}}
        order_book['bid']['price'] = list(self.price_map['bid'].keys())
        order_book['bid']['volume'] = [len(self.price_map['bid'][price]) for price in order_book['bid']['price']]
        order_book['ask']['price'] = list(self.price_map['ask'].keys())
        order_book['ask']['volume'] = [len(self.price_map['ask'][price]) for price in order_book['ask']['price']]
        return order_book
    
    def heat_map(self):
        N = 128
        lightness = 108
        reds = np.ones((N, 4))
        reds[:, 0] = np.linspace(1, 1, N)
        reds[:, 1] = np.linspace(lightness / N, 0, N)
        reds[:, 2] = np.linspace(lightness / N, 0, N)
        blues = np.ones((N, 4))
        blues[:, 0] = np.linspace(0, lightness / N, N)
        blues[:, 1] = np.linspace(0, lightness / N, N)
        blues[:, 2] = np.linspace(1, 1, N)
        newcolors = np.vstack([blues, reds])
        cmp = ListedColormap(newcolors)

        max_level = 4
        time = np.array(self.time_history)
        extended_time = []
        prices = []
        volumes = []
        for n in range(len(self.order_book_history)):
            # bid side 
            length = min(len(self.order_book_history[n]['bid']['price']), max_level)
            prices.extend(self.order_book_history[n]['bid']['price'][::-1][:max_level])
            volumes.extend([-1*x for x in self.order_book_history[n]['bid']['volume'][::-1][:max_level]])
            extended_time.extend(length*[time[n]])
            # ask side
            length = min(len(self.order_book_history[n]['ask']['price']), max_level)
            prices.extend(self.order_book_history[n]['ask']['price'][:max_level])
            volumes.extend(self.order_book_history[n]['ask']['volume'][:max_level])
            extended_time.extend(length*[time[n]])

        sc = plt.scatter(extended_time, prices, c=volumes, cmap=cmp, vmin=-5, vmax=5)
        plt.plot(time, self.ask_price_history, '-', color='black', linewidth=3)
        plt.plot(time, self.bid_price_history, '-', color='black', linewidth=3)
        ## plot micro price 
        ask = np.array(self.ask_price_history)
        bid = np.array(self.bid_price_history)
        ask_volume = np.array(self.ask_volume_history)
        bid_volume = np.array(self.bid_volume_history)
        micro_price = (ask*bid_volume + bid*ask_volume)/(ask_volume + bid_volume)
        plt.plot(time, micro_price, '-', color='grey', linewidth=2)
        ## plot market orders
        buy_market = np.array(self.ask_market_order_history)
        sell_market = np.array(self.bid_market_order_history)
        plt.scatter(time[buy_market==1], ask[buy_market==1], color='black', marker='x', s=80)
        plt.scatter(time[sell_market==1], bid[sell_market==1], color='black', marker='x', s=80)
        plt.xlim(left=time[0], right=time[-1])
        plt.colorbar(sc)
        plt.tight_layout()
        return None  



class Simulation(OrderBook):
    """
    - if config file exists, it contains the average shape of the book 
    and the parameters L, CR, MR. 

    - if it does not exists, we need to find the average shape of the book first, through a long simulation.
    we then save those paramaters to a pickle file. 

    Args:
        config (bool): if True, we use the config file to initialize the simulation.

    Returns:
        None
    """
    def __init__(self, config=False):
        super().__init__()
        if not config:
            # bunch of different configs
            # its hard to find a good trade off between volatility and queue length
            self.config = False
            # self.L = 20
            # self.LR = 1.0
            # self.MR = 2.0
            # self.CR = 1.0/30
            # Oomen paper config
            # self.L = 20
            # self.LR = 1 
            # self.MR = 10
            # self.CR = 0.2
            # EBAY config from Bouchaud's book 
            self.L = 20
            self.LR = 0.208 
            self.MR = 0.0209
            self.CR = 0.022
            # with this config prices are basically never moving at all 
            # Facebook config 
            # self.L = 20
            # self.LR = 0.169 
            # self.MR = 0.031
            # self.CR = 0.041
            # this gives depth of around 4 and some volatility, but its still very limited 
        else:
            # if the config is there, load it into an attribute of the class 
            with open('book_shape.pickle', 'rb') as f:
                self.config = pickle.load(f)
            self.L = self.config['config']['L']
            self.LR = self.config['config']['LR']
            self.MR = self.config['config']['MR']
            self.CR = self.config['config']['CR']
        self.rng = np.random.default_rng()

    def inititialize_order_book(self):
        if self.config: 
            depth = np.ceil(self.config['ask']).astype('int')
            for level in range(0,len(depth)):
                for _ in range(depth[level]):
                    order = {'type': 'limit', 'side': 'bid', 'price': 100-level}
                    self.limit_order(order)
                for _ in range(depth[level]):
                    order = {'type': 'limit', 'side': 'ask', 'price': 101+level}
                    self.limit_order(order)
        else:
            for price in range(101,121,1):    
                order = {'type': 'limit', 'side': 'ask', 'price': price}
                self.limit_order(order)
            for price in range(100,80,-1):    
                order = {'type': 'limit', 'side': 'bid', 'price': price}
                self.limit_order(order)
        self.init_time = self.time
        return None 

    def order_book_simulation(self, n_sim):
        for n in range(int(n_sim)):
            best_bid = self.get_best_bid()
            best_ask = self.get_best_ask()
            cr_bid = self.CR*len(self.order_map['bid']) 
            cr_ask = self.CR*len(self.order_map['ask'])
            probabilities = np.array([cr_bid, cr_ask, self.L*self.LR, self.L*self.LR, self.MR/2, self.MR/2], dtype=np.float64) 
            action_type, side = self.rng.choice([('cancel', 'bid'), ('cancel', 'ask'), ('limit', 'bid'), ('limit', 'ask'), ('market', 'bid'), ('market', 'ask')], p=probabilities/np.sum(probabilities))
            if action_type == 'limit':
                if side == 'bid':
                    price = self.rng.integers(best_ask-self.L, best_ask-1, endpoint=True)
                    order = {'type': action_type, 'side': side, 'price': price}
                    assert price < best_ask
                if side == 'ask':
                    price = self.rng.integers(best_bid+1, best_bid+self.L, endpoint=True)
                    order = {'type': action_type, 'side': side, 'price': price}
                    assert price > best_bid
                self.limit_order(order)        
            if action_type == 'cancel':
                id = self.rng.choice(list(self.order_map[side].keys()))
                order = {'type': action_type, 'side': side, 'id': id}
                self.cancellation(order)
            if action_type == 'market':
                order = {'type': action_type, 'side': side}
                self.market_order(order)
            if self.config: 
                if n%100 == 0:
                    self.log_limit_order_book_shape() 
            else:
                if n%100 == 0 and n > int(1e5):
                    self.log_limit_order_book_shape() 
            if n%int(1e5) == 0:
                print(n/int(1e5))
        return None  
    
    def average_book_shape(self):
        """
        - plots average book shape.
        - if there is no config file, it saves the average book shape and other simulation parameters to a pickle file.
        """
        book_shape = {'bid': [], 'ask': []}
        book_shape['config'] = {'L': self.L, 'LR': self.LR, 'MR': self.MR, 'CR': self.CR}
        book_shape['bid'] = np.mean(self.lob_shape_history['bid'], axis=0)
        book_shape['ask'] = np.mean(self.lob_shape_history['ask'], axis=0)
        print('avearage bid volumes')
        print(book_shape['bid'])
        print('average ask volumes')
        print(book_shape['ask'])
        plt.bar(range(-1,-21,-1), book_shape['bid'], color='red', label='bid')
        plt.bar(range(1,21,1), book_shape['ask'], color='blue', label='ask')
        plt.legend(loc='upper right')
        plt.xlabel('relative distance to mid price')
        plt.ylabel('average volume')
        if not self.config: 
            with open('book_shape.pickle', 'wb') as f:
                pickle.dump(book_shape, f)
        return None 

        

if __name__ == "__main__":
    Sim = Simulation(config=True)
    Sim.inititialize_order_book()
    Sim.order_book_simulation(n_sim=1e5)
    # Sim.average_book_shape()
    Sim.heat_map()
    # plt.figure()
    # Sim.mid_price_plot()
    # plt.figure()
    # Sim.average_book_shape()
    plt.show()


