from matplotlib import cm 
from collections import deque, namedtuple, OrderedDict
from operator import neg
import pickle 
from sortedcontainers import SortedDict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class OrderBook():
    """
    - simplified version of the order book. keeping track of the shape of the order book only
    - ask price is sorted dict in ascending order with pairs (price, volume)
    - bid price is sorted dict in descending order with pairs (price, volume)
    - the order book matches limit, market orders and cancellations
    """
    def __init__(self):
        self.price_map = {'bid': SortedDict(neg), 'ask': SortedDict()}
        self.time = 0 

    def process_order(self, order):
        assert order['type'] in ['limit', 'market', 'cancellation'], "order type not supported"
        
        if order['type'] == 'limit':
            self.limit_order(order)
        if order['type'] == 'market':
            self.market_order(order)
        if order['type'] == 'cancellation':
            self.cancellation(order)
        return None

    def limit_order(self, order):      
        """
        - if limit price is in price map, add volume to the price level
        - else create a new price level w
        ith the volume

        Args:
            order is dict with keys (type, side, price, volume)
        
        Returns:
            None. Changes the state of the order book internally. 
        """
        side = order['side']
        price = order['price']
        volume = order['volume']

        if side == 'ask':
            assert price > self.get_best_price('bid'), "ask price is lower than best bid price"
        if side == 'bid':
            assert price < self.get_best_price('ask'), "bid price is higher than best ask price"

        if price in self.price_map[side]:
            self.price_map[side][price] += volume 
        else:
            self.price_map[side].update({price: volume})
        self.time += 1
        return None 
    
    def market_order(self, order):
        """
        - match order against limt order in the book
        """

        side = order['side']
        market_volume = order['volume']

        if not self.price_map[side]:
            raise ValueError(f"{side} side is empty!")

        for price in self.price_map[side]: 
            limit_volume = self.price_map[side][price]
            if market_volume < limit_volume: 
                self.price_map[side][price] = limit_volume - market_volume
                market_volume = 0.0
                break 
            if market_volume == limit_volume:
                market_volume = 0.0 
                self.price_map[side].pop(price)
                break 
            if market_volume > limit_volume:
                market_volume = market_volume - limit_volume
                self.price_map[side].pop(price)

        if market_volume > 0.0:
            print(f"market order of size {market_volume} not fully executed!")

        return None
    
        ## analytics 

            
    def get_best_volumes(self, side, level=13):
        """
        - for ask side get volumes for the prices: best_bid+1, best_bid+2, ... , best_bid+level
        - same logic for the bid side 
        """
        if side == 'ask':
            opposite_best_price = self.price_map['bid'].peekitem(index=0)[0]
            best_prices = np.arange(opposite_best_price+1, opposite_best_price+level+1, 1)
        if side == 'bid':
            opposite_best_price = self.price_map['ask'].peekitem(index=0)[0]
            best_prices = np.arange(opposite_best_price-1, opposite_best_price-level-1, -1)
        volumes = [self.price_map[side][price] if price in self.price_map[side] else 0 for price in best_prices]
        return np.array(volumes)


    def cancellation(self, order):
        """
        - remove volume from the specified price level
        - cap cancellation volume at the volume in the price level
        """
        # TODO: write assertion debugging function 
        side = order['side']
        volume = order['volume']
        price = order['price']
        assert self.price_map[side][price], "price level to cancel does not exist"
        assert volume >= 0, "cancellation volume is negative"
        volume = max(self.price_map[side][price] - volume, 0.0)
        if volume == 0.0:
            self.price_map[side].pop(price)
        elif volume > 0.0:
            self.price_map[side][price] = volume
        return None

    def get_best_price(self, side):
        return self.price_map[side].peekitem(index=0)[0]


class Simulation():
    """
    attributes: 

    - order book 
    - cancellation intensities 

    Args:
        config (bool): if True, we use the config file to initialize the simulation.
        
    Returns:
        None
    """

    def __init__(self):
        self.rng = np.random.default_rng()
        self.order_book = OrderBook()

        # only up to the 13th level, zero afterwards 
        self.limit_intensities = np.array([0.2842, 0.5255, 0.2971, 0.2307, 0.0826, 0.0682, 0.0631, 0.0481, 0.0462, 0.0321, 0.0178, 0.0015, 0.0001])
        self.limit_intensities = np.pad(self.limit_intensities, (0,30-len(self.limit_intensities)), 'constant', constant_values=(0))
        self.limit_intensity = np.sum(self.limit_intensities)
        self.cancel_intensities = 1e-3*np.array([0.8636, 0.4635, 0.1487, 0.1096, 0.0402, 0.0341, 0.0311, 0.0237, 0.0233, 0.0178, 0.0127, 0.0012, 0.0001])
        self.cancel_intensities = np.pad(self.cancel_intensities, (0,30-len(self.cancel_intensities)), 'constant', constant_values=(0))
        self.market_intesity = 0.1237

        # self.market_intesity = 2.0
        self.initial_shape = np.array([276, 1129, 1896, 1924, 1951, 1966, 1873, 1786, 1752, 1691, 1558, 1435, 1338, 1238, 1122, 1036, 943, 850, 796, 716, 667, 621, 
                              560, 490, 443, 400, 357, 317, 285, 249]) 
        
        # lognormal distribution parameters 
        self.market_volume_parameters = {'mean':4.00, 'sigma': 1.19} 
        self.limit_volume_parameters = {'mean':4.47, 'sigma': 0.83}
        self.cancel_volume_parameters = {'mean':4.48, 'sigma': 0.82}


        self.bid_prices = []
        self.ask_prices = []
        self.bid_volumes = []
        self.ask_volumes = []
        self.order_book_history = []
        self.ask_trades = []
        self.bid_trades = []

        self.order = None 

        self.lob_shape_history = {'bid': [], 'ask': []}

    def initialize(self):
        # iniitial shape of the order book 
        for idx, price in enumerate(np.arange(1000, 1000-30, -1)):
            self.order_book.price_map['bid'].update({price: self.initial_shape[idx]})
        for idx, price in enumerate(np.arange(1001, 1000+31, 1)): 
            self.order_book.price_map['ask'].update({price: self.initial_shape[idx]})
        # new initial shape of the order book
        for idx, price in enumerate(np.arange(1000, 1000-30, -1)):
            self.order_book.price_map['bid'].update({price: 1000})
        for idx, price in enumerate(np.arange(1001, 1000+31, 1)): 
            self.order_book.price_map['ask'].update({price: 1000})
        # new initial shape 
        # this shape was created from the initial shape by adding 1000 to each price level
        # and boundary conditions 250 for levels 30 to 60 
        shape = [370.86103498, 1082.79416938, 1641.18126655, 1929.85272458, 1985.4760373,
 1984.59506303, 1953.01178331, 1899.57172227, 1810.03489214, 1690.87559764,
 1594.33200374, 1498.44755192, 1434.21632838, 1348.5491408,  1255.67333407,
 1166.43990325, 1073.5007203,   995.06602917,  938.93919677,  878.53068957,
  835.43277174,  806.94020008,  764.01787325,  708.89343769,  653.58418488,
  572.98503493,  480.226725,    401.27687152,  338.74447331,  281.79973522]
        # iniitial shape of the order book 
        for idx, price in enumerate(np.arange(1000, 1000-30, -1)):
            self.order_book.price_map['bid'].update({price: shape[idx]})
            self.order_book.price_map['ask'].update({price: shape[idx]})


    def get_best_volumes(self, side, level=30):
        """
        """
        if side == 'ask':
            opposite_best_price = self.order_book.price_map['bid'].peekitem(index=0)[0]
            best_prices = np.arange(opposite_best_price+1, opposite_best_price+level+1, 1)
        if side == 'bid':
            opposite_best_price = self.order_book.price_map['ask'].peekitem(index=0)[0]
            best_prices = np.arange(opposite_best_price-1, opposite_best_price-level-1, -1)
        volumes = [self.order_book.price_map[side][price] if price in self.order_book.price_map[side] else 0 for price in best_prices]
        return np.array(volumes)

    def get_best_price(self, side):
        return self.order_book.price_map[side].peekitem(index=0)[0]
    
    def get_best_volume(self, side):
        return self.order_book.price_map[side].peekitem(index=0)[1]
    
    # def order_book_to_dict(self):        
    def simulate_order(self): 
        # bid and ask price
        best_bid = self.get_best_price('bid')
        best_ask = self.get_best_price('ask')
        # bid and ask volumes
        bid_volumes = self.get_best_volumes('bid')        
        ask_volumes = self.get_best_volumes('ask')
        # ask cancel intensities 
        ask_cancel_intensity = np.sum(self.cancel_intensities*ask_volumes)
        bid_cancel_intensity = np.sum(self.cancel_intensities*bid_volumes) 
        # draw from the distribution
        probability = np.array([self.market_intesity, self.market_intesity, self.limit_intensity, self.limit_intensity, bid_cancel_intensity, ask_cancel_intensity])
        probability = probability/np.sum(probability)
        action, side = self.rng.choice([('market', 'bid'), ('market', 'ask'), ('limit', 'bid'), ('limit', 'ask'), ('cancellation', 'bid'), ('cancellation', 'ask')], p=probability)
        # print(action, side)
        if action == 'limit':
            probability = self.limit_intensities/np.sum(self.limit_intensities)
            level = self.rng.choice(np.arange(1, 30+1, 1), p=probability)
            if side == 'bid':
                # TODO: write attribute to LOB which gets the best bid/ask. dependent property or something like that. 
                price = self.get_best_price('ask') - level 
            if side == 'ask':
                price = self.get_best_price('bid') + level 
            volume = self.rng.lognormal(mean=self.limit_volume_parameters['mean'], sigma=self.limit_volume_parameters['sigma'])
            order = {'type': action, 'side': side, 'price': price, 'volume': volume}
            # print(f'limit order: {side}, {level}')
        if action == 'market':
            volume = self.rng.lognormal(mean=self.market_volume_parameters['mean'], sigma=self.market_volume_parameters['sigma'])
            order = {'type': action, 'side': side, 'price': None, 'volume': volume}
            # print(f'market order: {side}')
        if action == 'cancellation':
            if side == 'ask':
                probability = ask_volumes*self.cancel_intensities
                probability = probability/np.sum(probability)
                level = self.rng.choice(np.arange(1,30+1,1), p=probability) 
                price = self.get_best_price('bid') + level 
            if side == 'bid':
                probability = bid_volumes*self.cancel_intensities
                probability = probability/np.sum(probability)
                level = self.rng.choice(np.arange(1,30+1,1), p=probability) 
                price = self.get_best_price('ask') - level
            volume = self.rng.lognormal(mean=self.cancel_volume_parameters['mean'], sigma=self.cancel_volume_parameters['sigma'])
            order = {'type': action, 'side': side, 'price': price, 'volume': volume}
            # print(f'cancel order: {side}, {level}')
        return order 
    
    def cancel_far_out_orders(self):
        best_bid = self.get_best_price('bid')
        best_ask = self.get_best_price('ask')

        # delete empty price levels and set boundary condition 
        for price in self.order_book.price_map['bid'].keys():
            if price < best_ask - 30:
                self.order_book.price_map['bid'].pop(price)
        for price in np.arange(best_ask-31, best_ask-61, -1):
            self.order_book.price_map['bid'][price] = 250
        for price in self.order_book.price_map['ask'].keys():
            if price > best_bid + 30:
                self.order_book.price_map['ask'].pop(price)
        for price in np.arange(best_bid+31, best_bid+61, +1):
            self.order_book.price_map['ask'][price] = 250

    def logging(self, order):
        if order == None:
            self.bid_trades.append(0)
            self.ask_trades.append(0)
        elif order['type'] == 'market':
            if order['side'] == 'bid':
                self.bid_trades.append(1)
                self.ask_trades.append(0)
            if order['side'] == 'ask':
                self.bid_trades.append(0)
                self.ask_trades.append(1)
        else:
            self.bid_trades.append(0)
            self.ask_trades.append(0)            
        self.ask_prices.append(self.get_best_price('ask'))
        self.bid_prices.append(self.get_best_price('bid'))
        self.bid_volumes.append(self.get_best_volume('bid'))
        self.ask_volumes.append(self.get_best_volume('ask'))
        return None 
    
    def plot_bid_ask(self):
        ask = np.array(self.ask_prices)
        bid = np.array(self.bid_prices)
        ask_volume = np.array(self.ask_volumes)
        bid_volume = np.array(self.bid_volumes)
        micro_price = (ask*bid_volume + bid*ask_volume)/(ask_volume + bid_volume)
        time = np.arange(0, len(ask), 1)
        buy_market = np.array(self.ask_trades)
        sell_market = np.array(self.bid_trades)
        plt.plot(time, bid, '-', color='grey', label='bid/ask')
        plt.plot(time, ask, '-', color='grey')
        plt.plot(time, micro_price, '-', color='blue', label='micro price')
        plt.scatter(time[buy_market==1], ask[buy_market==1], color='red', marker='x')
        plt.scatter(time[sell_market==1], bid[sell_market==1], color='red', marker='x', label='market order')
        # plt.xlim(left=time[0], right=time[-1])
        plt.legend(loc='best')
        # plt.xlabel('tick time')
        # plt.ylabel('relative price')
        # plt.plot(self.ask_prices)
        # plt.plot(self.bid_prices)
        return None

    def log_limit_order_book_shape(self):
        bid_volumes = self.get_best_volumes(level=30, side = 'bid')
        ask_volumes = self.get_best_volumes(level=30, side = 'ask')
        self.lob_shape_history['bid'].append(bid_volumes)
        self.lob_shape_history['ask'].append(ask_volumes)
        return None
    
    def average_book_shape(self):
        """
        - plots average book shape.
        - if there is no config file, it saves the average book shape and other simulation parameters to a pickle file.
        """
        book_shape = {'bid': [], 'ask': []}
        # book_shape['config'] = {'L': self.L, 'LR': self.LR, 'MR': self.MR, 'CR': self.CR}
        book_shape['bid'] = np.mean(self.lob_shape_history['bid'], axis=0)
        book_shape['ask'] = np.mean(self.lob_shape_history['ask'], axis=0)
        print('avearage bid volumes')
        print(book_shape['bid'])
        print('average ask volumes')
        print(book_shape['ask'])
        plt.bar(range(0,-30,-1), book_shape['bid'], color='red', label='bid')
        plt.bar(range(1,31,1), book_shape['ask'], color='blue', label='ask')
        plt.legend(loc='upper right')
        plt.xlabel('relative distance to mid price')
        plt.ylabel('average volume')
        return None 
    

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

        max_level = 10
        N = len(self.lob_shape_history['bid'])
        time = np.arange(N)
        extended_time = []
        prices = []
        volumes = []
        for n in range(N):
            # bid side 
            prices.extend(list(range(self.ask_prices[n]-1, self.ask_prices[n] - max_level - 1, -1)))
            volumes.extend([-1*x for x in self.lob_shape_history['bid'][n][:max_level]])
            extended_time.extend(max_level*[time[n]])
            # ask side
            prices.extend(list(range(self.bid_prices[n]+1, self.bid_prices[n] + max_level + 1, 1)))
            volumes.extend(self.lob_shape_history['ask'][n][:max_level])
            extended_time.extend(max_level*[time[n]])

        sc = plt.scatter(extended_time, prices, c=volumes, cmap=cm.seismic, vmin=-2000, vmax=2000)
        # sc = plt.scatter(extended_time, prices, c=volumes, cmap=cmp, vmin=-2000, vmax=2000)
        plt.plot(time, self.ask_prices, '-', color='black', linewidth=3)
        plt.plot(time, self.bid_prices, '-', color='black', linewidth=3)
        ## plot micro price 
        ask = np.array(self.ask_prices)
        bid = np.array(self.bid_prices)
        # ask_volume = np.array(self.ask_volumes)
        # bid_volume = np.array(self.bid_volumes)
        # micro_price = (ask*bid_volume + bid*ask_volume)/(ask_volume + bid_volume)
        # plt.plot(time, micro_price, '-', color='grey', linewidth=2)
        ## plot market orders
        buy_market = np.array(self.ask_trades)
        sell_market = np.array(self.bid_trades)
        plt.scatter(time[buy_market==1], ask[buy_market==1], color='black', marker='x', s=80)
        plt.scatter(time[sell_market==1], bid[sell_market==1], color='black', marker='x', s=80)
        plt.xlim(left=time[0], right=time[-1])
        plt.colorbar(sc)
        plt.tight_layout()
        return None  





if __name__ == "__main__":
    for _ in range(5):
        Sim = Simulation()
        Sim.initialize()
        print(Sim.get_best_volumes('bid'))
        print(Sim.get_best_volumes('ask'))
        for n in range(int(2e3)):
            order = Sim.simulate_order()
            Sim.logging(order)
            Sim.cancel_far_out_orders()
            # if n%100 == 0 and n > 5e5:
            Sim.log_limit_order_book_shape()
            # if n%100000 == 0:
                # print(n)
            Sim.order_book.process_order(order)
            # print(Sim.get_best_volumes('bid'))
            # print(Sim.get_best_volumes('ask'))
        plt.figure()
        Sim.plot_bid_ask()      
        plt.figure()
        Sim.average_book_shape()
        plt.figure()
        Sim.heat_map()
        print('bid volumes')
        print(Sim.get_best_volumes('bid'))
        print('ask volumes')
        print(Sim.get_best_volumes('ask'))
    plt.show()      
    # ToDo: volumes is increasing, why


