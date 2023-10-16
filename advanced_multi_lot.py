from copy import deepcopy
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

def action_to_n_orders(action, volume):
    available_volume = volume
    target_n_orders = []
    for l in range(len(action)):
        # starts at zero, runs up to self.n_levels+1
        # [0,1,...,self.n_levels+1]: first one if for market order, last one is for inactive volume
        # np.round rounds values in [0,0.5] to 0, and values in [0.5,1] to 1 
        n_orders_on_level = min(np.round(action[l]*volume).astype(int), available_volume)
        available_volume -= n_orders_on_level
        target_n_orders.append(n_orders_on_level) 
    target_n_orders[-1] += available_volume
    assert available_volume >= 0 
    assert sum(target_n_orders) == volume
    return target_n_orders

class PeggingAgent:
    # observation space = action space = [-1,0,1,2,3,4] corresponding to market order and levels 
    def __init__(self) -> None:
        pass

    def get_action(self, obs : np.ndarray, level=0) -> int:
        # always move the limit order to the best price 
        return 0

class SubmitAndLeaveAgent:
    def __init__(self, level=2) -> None:
        self.level = level
        pass

    def get_action(self, obs : np.ndarray) -> int:
        # at time 0 place order at desired level 
        if obs[0] == 0:
            return self.level
        else:
        # otherwise leave the order where it is 
            return obs[1] 

class Market(gym.Env):

    def __init__(self, config):

        # global initializations        
        if config['env_type'] == 'simple':
            self.imbalance_trader = False
            self.drift_down = False
        elif config['env_type'] == 'imbalance':
            self.imbalance_trader = True
            self.drift_down = False
        elif config['env_type'] == 'down':
            self.imbalance_trader = True
            self.drift_down = True
        else:
            raise ValueError('env_type not supported')
        
        self.np_random = np.random.default_rng(config['seed'])
        self.market_agent_id = 'market_agent'
        self.agent_id = 'agent'
        self.initial_volume = config['initial_volume']
        self.total_n_steps = config['total_n_steps']

        # market related attributes
        self.limit_intensities = np.array([0.2842, 0.5255, 0.2971, 0.2307, 0.0826, 0.0682, 0.0631, 0.0481, 0.0462, 0.0321, 0.0178, 0.0015, 0.0001])
        self.limit_intensities = np.pad(self.limit_intensities, (0,30-len(self.limit_intensities)), 'constant', constant_values=(0))
        self.cancel_intensities = 1e-3*np.array([0.8636, 0.4635, 0.1487, 0.1096, 0.0402, 0.0341, 0.0311, 0.0237, 0.0233, 0.0178, 0.0127, 0.0012, 0.0001])
        self.cancel_intensities = np.pad(self.cancel_intensities, (0,30-len(self.cancel_intensities)), 'constant', constant_values=(0))
        self.market_intesity = 0.1237
        # TODO: add run configs at the top of the file 
        if config['ada']:
            shape = np.load('/u/weim/lob/data/stationary_shape.npz')
        else:
            shape = np.load('/Users/weim/projects/lob/data/stationary_shape.npz')
        self.initial_shape = np.mean([shape['bid'], shape['ask']], axis=0)
        self.initial_shape = np.rint(self.initial_shape).astype(int)

        # lognormal distribution parameters 
        self.market_volume_parameters = {'mean':4.00, 'sigma': 1.19} 
        self.limit_volume_parameters = {'mean':4.47, 'sigma': 0.83}
        self.cancel_volume_parameters = {'mean':4.48, 'sigma': 0.82}

        # observation space, action space 
        # 4 features: time, volume, price drift, imbalance , 
        # 5 allocations: [1,2,3,4:overflow, 5:inactive], 
        # 6 volumes [1,2,3] bid/ask, total = 4+5+6 = 15
        # self.n_levels describes on which levels we place, currently pace on 3 levels 
        self.n_levels = 3 
        self.n_actions = self.n_levels+2
        n_observations = 4+(self.n_levels+2)+2*(self.n_levels) #=4+4+6=15
        assert n_observations == 15
        self.observation_space = Box(low=-1, high=1, shape=(n_observations,)) 
        self.action_space = Box(low=-10, high=10, shape=(self.n_levels+2,), dtype=np.float32, seed=config['seed'])
        # self.action_space = Simplex(shape=(4,), concentration=np.array([1] * 4), dtype=np.float32)
        self.log = config['log']

        

    ## order book related methods 
    def process_order(self, order):
        """
        - an order is a dictionary with fields agent_id, type, side, price, volume, order_id
        - some of those fields are optional depending on the order type 
        """

        # all orders should have a type and an agent_id
        assert order['type'] in ['limit', 'market', 'cancellation', 'modification'], "order type not supported"
        assert order['agent_id'] in [self.market_agent_id, self.agent_id]
 
        if order['type'] == 'limit':            
            # limit order needs side, price, volume 
            assert order['side'] in ['bid', 'ask'], "side not specified"
            # could enforce integer volume. maybe not necessary. 
            # assert isinstance(order['volume'], np.int64), "volume not integer"
            assert order['volume'] > 0, "volume not positive"
            assert order['price'], "price not specified"
            order['order_id'] = self.update_n
        
        if order['type'] == 'market':
            # market order needs side, volume 
            assert order['side'] in ['bid', 'ask'], "side not specified"
            # assert isinstance(order['volume'], np.int64), "volume not integer"
            assert order['volume'] > 0, "volume not positive"
            # market order does not order_id and price 
                
        if order['type'] == 'cancellation':
            # cancellation needs order id and nothing else 
            # assert isinstance(order['order_id'], np.int64), "order id not integer"
            assert order['order_id'] >= 0, "not order id specified"
        
        if order['type'] == 'modification':
            # volume indicates the new volume of the order with order_id
            # assert isinstance(order['order_id'], np.int64), "order id not integer"
            # assert isinstance(order['volume'], np.int64), "volume not integer"
            assert order['volume'] > 0, "volume not positive"
            assert order['volume'] <= self.order_map[order['order_id']]['volume'], "new volume larger than original order volume"

        if order['type'] == 'limit':
            msg = self.limit_order(order)
        if order['type'] == 'market':
            msg = self.market_order(order)
        if order['type'] == 'cancellation':
            msg = self.cancellation(order)
        if order['type'] == 'modification':
            msg = self.modification(order)

        # ToDo: Implement modification of orders to reduce volume. (adding volume is just a new limit order at the
        # back of the queue)

        self.update_n += 1 
        return msg

    def send_limit_message(self, order):
        assert order['type'] == 'limit'
        return order['order_id']              

    def send_market_message(self, order, average_price, filled_volume):
        assert order['type'] == 'market'
        return average_price, filled_volume

    def send_fill_message(self, fill_list):
        return fill_list

    def limit_order(self, order):      
        """
        - if limit price is in price map, add volume to the price level
        - else create a new price level with the volume

        Args:
            order is dict with keys (type, side, price, volume)
        
        Returns:
            None. Changes the state of the order book internally
                - "order_id" is assigned to the order
                - limit order is added to the order map under the key "order_id" and with the whole dict order as value 
                - limit order is added to the price map under the key "price" and with order_id as value
        """

        order_id = order['order_id']
        side = order['side']
        price = order['price']
        volume = order['volume']

        # only do this check if the opposite side is not empty
        if side == 'ask' and self.price_map['bid']:
            assert price > self.get_best_price('bid'), "sent ask limit order with price <= bid price"
        if side == 'bid' and self.price_map['ask']:    
            assert price < self.get_best_price('ask'), "sent bid limit order with price >= ask price"

        if price in self.price_map[side]:
            # add order to price level 
            self.price_map[side][price].add(order_id) 
        else:
            # SortedList 
            self.price_map[side][price] = SortedList([order_id])
        
        self.order_map[order_id] = order

        return ('limit', order_id) 

    def market_order(self, order):
        """
        - match order against limt order in the book
        - return profit message to both agents 
        - modify the state of filled orders in the book 
        """

        side = order['side']
        market_volume = order['volume']

        if not self.price_map[side]:
            raise ValueError(f"{side} side is empty!")

        average_price = 0.0
        
        filled_orders = {self.agent_id:[], self.market_agent_id:[]}

        prices = list(self.price_map[side].keys())
        for price in prices: 
            # cp = counterparty 
            cp_order_ids = deepcopy(self.price_map[side][price])
            for cp_order_id in cp_order_ids:
                cp_order = self.order_map[cp_order_id]
                cp_agent_id = cp_order['agent_id']
                if market_volume < cp_order['volume']:
                    # this updates volume in the order map 
                    cp_order['volume'] -= market_volume
                    # add additional fields for information, these fields will also be carried in the order map
                    cp_order['old_volume'] = market_volume + cp_order['volume']
                    cp_order['partial_fill'] = True
                    filled_orders[cp_agent_id].append(cp_order)
                    average_price += price * market_volume
                    # self.order_map[cp_order_id]['volume'] -= market_volume, this is not necessary, opposite order is a reference to opposite_order['volume']
                    market_volume = 0.0
                    break
                if market_volume == cp_order['volume']:
                    filled_orders[cp_agent_id].append(cp_order)  
                    average_price += price * market_volume
                    self.price_map[side][price].remove(cp_order_id)
                    self.order_map.pop(cp_order_id)
                    market_volume = 0.0
                    break
                if market_volume > cp_order['volume']:
                    filled_orders[cp_agent_id].append(cp_order)
                    average_price += price * cp_order['volume']
                    self.price_map[side][price].remove(cp_order_id)              
                    self.order_map.pop(cp_order_id)      
                    market_volume = market_volume - cp_order['volume']
            if not self.price_map[side][price]:
                self.price_map[side].pop(price)
            if market_volume == 0.0:
                break


        if market_volume > 0.0:
            print(f"market order of size {order['volume']} not fully executed, {market_volume} remaining!")            
        
        filled_volume = order['volume'] - market_volume

        # market: return type, filled orders, average price
        # limit: return order_id
        # cancellation: return ids of cancelled orders          
        return 'market', filled_orders, average_price, order
    
    def cancellation(self, order):
        order_id = order['order_id']
        order = self.order_map[order_id]
        side = order['side']
        price = order['price']
        self.price_map[side][price].remove(order_id)
        self.order_map.pop(order_id)
        # not {} = True 
        if not self.price_map[side][price]:
            self.price_map[side].pop(price)
        return 'cancellation', order_id
    
    def modification(self, order):
        order_id = order['order_id']
        self.order_map[order_id]['volume'] = order['volume']
        return 'modification', order_id

    ## simulation related methods

    def initialize_book(self):  
        # TODO: 30 as a parameter 
        # order_list = []
        L = 30

        for idx, price in enumerate(np.arange(1000, 1000-L, -1)):
            order = {'agent_id': self.market_agent_id, 'type': 'limit', 'side': 'bid', 'price': price, 'volume': self.initial_shape[idx]}
            self.process_order(order)
        for idx, price in enumerate(np.arange(1001, 1000+L+1, 1)): 
            order = {'agent_id': self.market_agent_id, 'type': 'limit', 'side': 'ask', 'price': price, 'volume': self.initial_shape[idx]}
            self.process_order(order)
        return None 

    def generate_order(self):
        """

        The method generates orders as follows:
        Update cancel intensity. Cancel intensities are scaled according to the volume in the book. 
        Draw event type (limit, market, cancel) according to intensities.
        Draw volume according to lognormal distribution.
        Draw side with 50% probability.
        For cancellation and limit, draw price according to intensities.          

        Args:
            best bid and ask price
            bid and ask volumes they should match the length of the limit, cancel intensities 

        Note: the method directly cancells the orders in the book. or adds limit orders. or executes market orders. 
        It doesnt only generate orders. But also matches them against the order book. 
            
        Returns:
        limit, market, or cancellation 
            limit example: order = {'agent_id': 'agent', 'type': 'limit', 'side': , 'price': 100, 'volume': 10.0}
            market example: order = {'agent_id': 'agent', 'type': 'market', 'side': , 'volume': 10.0}
            cancellation: 
                is a list of cancellations 
                for given volume v, find first order id such that the sum of ids is larger than v 
                this might lead to more cancellations than volume v 
                TODO: add modification of orders to get exactly volume v
        """

        L = len(self.limit_intensities)

        _, ask_volumes = self.level2(side='ask', level=L)
        _, bid_volumes = self.level2(side='bid', level=L)
        best_ask = self.get_best_price(side='ask')
        best_bid = self.get_best_price(side='bid') 

        ask_cancel_intensity = np.sum(self.cancel_intensities*ask_volumes)
        bid_cancel_intensity = np.sum(self.cancel_intensities*bid_volumes)
        limit_intensity = np.sum(self.limit_intensities)

        # check if bid ask volumes are zero at the same time 
        if (bid_volumes[0] == 0) and (ask_volumes[0] == 0):
            imbalance = 0
        else:
            imbalance = ((bid_volumes[0]) - ask_volumes[0])/(bid_volumes[0] + ask_volumes[0])
            # imbalance = (np.sum(bid_volumes[:1]) - np.sum(ask_volumes[:1]))/(np.sum(bid_volumes[:1]) + np.sum(ask_volumes[:1]))    
        if np.isnan(imbalance):
            print(imbalance)
            print(bid_volumes)
            print(ask_volumes)
            raise ValueError('imbalance is nan')
        if self.imbalance_trader:
            # market intensity 
            market_buy_intensity = self.market_intesity*(1+imbalance)
            market_sell_intensity = self.market_intesity*(1-imbalance)
            # test 
            # market_buy_intensity = self.market_intesity*(1-imbalance)
            # market_sell_intensity = self.market_intesity*(1+imbalance)
        else:
            market_buy_intensity = self.market_intesity
            market_sell_intensity = self.market_intesity

        probability = np.array([market_sell_intensity, market_buy_intensity, limit_intensity, limit_intensity, bid_cancel_intensity, ask_cancel_intensity])
        # market sell order arrives on the best bid 
        # market buy order arrives on the best ask
        probability /= np.sum(probability)        

        action, side = self.np_random.choice([('market', 'bid'), ('market', 'ask'), ('limit', 'bid'), ('limit', 'ask'), ('cancellation', 'bid'), ('cancellation', 'ask')], p=probability)

        if action == 'limit':
            probability = self.limit_intensities/np.sum(self.limit_intensities)
            level = self.np_random.choice(np.arange(1, L+1, 1), p=probability)
            if side == 'bid':
                price = best_ask - level 
            if side == 'ask':
                price = best_bid + level 
            volume = 0
            while volume == 0:
                volume = self.np_random.lognormal(mean=self.limit_volume_parameters['mean'], sigma=self.limit_volume_parameters['sigma'])
                volume = np.rint(volume).astype(int)
            order = {'agent_id': self.market_agent_id, 'type': action, 'side': side, 'price': price, 'volume': volume}

        if action == 'market':
            volume = 0 
            while volume == 0:
                volume = self.np_random.lognormal(mean=self.market_volume_parameters['mean'], sigma=self.market_volume_parameters['sigma'])
                volume = np.rint(volume).astype(int)
            order = {'agent_id': self.market_agent_id, 'type': action, 'side': side, 'volume': volume}
        
        if action == 'cancellation':
            if side == 'ask':
                probability = ask_volumes*self.cancel_intensities
                probability /= np.sum(probability)
                level = self.np_random.choice(np.arange(1,L+1,1), p=probability) 
                price = best_bid + level 
            if side == 'bid':
                probability = bid_volumes*self.cancel_intensities
                probability = probability/np.sum(probability)
                level = self.np_random.choice(np.arange(1,L+1,1), p=probability) 
                price = best_ask - level
            volume = 0 
            while volume == 0:
                volume = self.np_random.lognormal(mean=self.cancel_volume_parameters['mean'], sigma=self.cancel_volume_parameters['sigma'])
                volume = np.rint(volume).astype(int)
            order = {'agent_id': self.market_agent_id, 'type': action, 'side': side, 'price': price, 'volume': volume}
        
        if action == 'limit' or action == 'market':
            return self.process_order(order)
        else:
            return self.find_orders_to_cancel(side=order['side'], cancel_volume=order['volume'], price=order['price'])

    def find_orders_to_cancel(self, side, cancel_volume, price):
        # TODO: order volumes should always be integer 
        # careful here. only cancel the market agent orders. not any other orders 
        level = self.price_map[side][price].copy()
        # modify cancellation volume 
        for cp_order_id in level[::-1]:
            cp_order = self.order_map[cp_order_id]
            assert cp_order['agent_id'] in [self.agent_id, self.market_agent_id]
            if cp_order['agent_id'] == self.agent_id:
                # do not cancel orders generated by the agent 
                # only those by the market agent
                continue
            if cancel_volume < cp_order['volume']:
                # modify cp order volume 
                new_volume = cp_order['volume'] - cancel_volume
                cancel_volume = 0
                self.process_order({'agent_id': self.agent_id, 'type': 'modification', 'order_id': cp_order_id, 'volume': new_volume})
            if cancel_volume >= cp_order['volume']:
                # modify cancellation volume 
                cancel_volume -= cp_order['volume']
                self.process_order({'agent_id': self.agent_id,  'type': 'cancellation', 'order_id': cp_order_id})
            if cancel_volume == 0:
                break
        
        # cancellation by the market agent 
        return 'cancellation_market', None

    def cancel_far_out_orders(self,L=30):
        best_bid = self.get_best_price(side='bid')
        best_ask = self.get_best_price(side='ask')

        for price in self.price_map['bid'].keys():
            if price < best_ask - L:
                for order_id in self.price_map['bid'][price]:
                    if self.order_map[order_id]['agent_id'] == self.market_agent_id:
                        self.process_order({'agent_id': self.agent_id, 'type': 'cancellation', 'order_id': order_id})

        for price in self.price_map['ask'].keys():
            if price > best_bid + L:
                for order_id in self.price_map['ask'][price]:
                    if self.order_map[order_id]['agent_id'] == self.market_agent_id:
                        self.process_order({'agent_id': self.agent_id, 'type': 'cancellation', 'order_id': order_id})
        
        return None 
    

    ## analytics methods 

    def find_queue_position(self, order_id, all_levels=True):        
        # implement all levels option 
        order = self.order_map[order_id]        
        side = order['side']
        queue_position = 0
        for price in self.price_map[side]:
            level = self.price_map[side][price]
            for id in level:
                if id == order_id:
                    return queue_position
                queue_position += self.order_map[id]['volume']
        raise ValueError('order_id not found on this side of the book')

    def get_best_price(self, side):
        return self.price_map[side].keys()[0]
    
    def level2(self, side, level=10):
        """
        output: 
            if side == 'bid' the output is a tuple of two np arrays:
            - best bid prices up to level: [p_1, p_2, ... , p_level]
            - np array of best bid volumes up to level: [v_1, v_2, ... , v_level]
            - includes empty price levels
        """        
        assert side in ['bid', 'ask'], "side must be either bid or ask"

        if side == 'bid':
            prices = np.arange(self.get_best_price('ask')-1, self.get_best_price('ask')-level-1, -1)
        if side == 'ask':
            prices = np.arange(self.get_best_price('bid')+1, self.get_best_price('bid')+level+1, 1)

        volumes = []
        for price in prices:
            v = 0 
            if price in self.price_map[side]:
                for order_id in self.price_map[side][price]:
                    v += self.order_map[order_id]['volume']
            volumes.append(v)
        
        volumes = np.array(volumes)

        return prices, volumes

    def level1(self, side):
        """
        output: (best_price, best_volume)
        """

        best_price = self.get_best_price(side)
        best_volume = 0
        for order_id in self.price_map[side][best_price]:
            best_volume += self.order_map[order_id]['volume']

        return best_price, best_volume 


    ## gym methods 

    def reset(self, seed=None, options=None, initialize_orders=True):
        """
        - initialize order and price map, set of active orders
        - reset time and update_n to zero 
        - reset volume to initial volume         
        - reset log variables 
        """

        super().reset(seed=seed)

        self.order_map = {}
        self.price_map = {'bid': SortedDict(neg), 'ask': SortedDict()}
        self.update_n = 0 
        self.time = 0 

        # order information 
        self.active_volume = 0
        self.inactive_volume = self.initial_volume
        self.volume = self.initial_volume
        self.active_orders = set()
        self.active_orders_within_first_levels = set()
        # n orders describes the number of the agents orders on each level. 
        # TODO: write this into a function called get_order_info. 
        # the function should summarize all the information on the current agents orders. location, queue position. 
        self.n_orders = []        
        assert self.volume > 0 
        self.initialize_book()

        # update_n increases every time the the order book is updated 
        # not necessarily the same as time 
        # there might be multiple updates at the same time (example: several orders get cancelled at the same time, several orders are filled by a market order) 
        # actions submitted by a single agent all happen at the same time, sequentially  
        # time updates after agent order is process or after market order is process 
    
        # n levels for ation and observation space 
        # levels: 0:market order, 1:first level, ..., 3-th level, 4: inactive level 
        # shape of levels 1, to, 3


        ## logging 
        if self.log:
            self.bid_volumes = []
            self.ask_volumes = []
            self.bid_prices = []
            self.ask_prices = []
            self.trades = []

            self.best_bid_prices = []
            self.best_ask_prices = []
            self.best_bid_volumes = []
            self.best_ask_volumes = []
                
            self.initial_ask = self.get_best_price(side='ask')
            self.initial_bid = self.get_best_price(side='bid')
            self.initial_mid_price = (self.initial_ask + self.initial_bid)/2

        
        # best_ask = self.get_best_price(side='ask')
        # if initialize_orders:
        #     for _ in range(self.volume):
        #         order = {'agent_id':self.agent_id , 'type':'limit', 'side':'ask', 'price':best_ask+self.initial_level, 'volume':1} 
        #         out = self.process_order(order)
        #         self.active_orders.add(out[1])
        # else:
        #     pass
        # self.active_order = out[1]
        # self.level = level


        return self._get_obs(), {}
    
    def _get_reward(self, reward, traded_volume):
        return 1000*(reward - self.initial_bid*traded_volume)/(self.initial_volume*self.initial_bid)

    def _get_agents_order_distribution(self):
        """
        - find order distribution of currently active orders
        - that is the distribution of orders on levels [1, 2, ..., self.n_levels]
        - e.g self.n_levels = 3
            - volume = 10 
            - action space is [a0=market, a1, a2, a3, a4=inactive]
            - observation is [o1, o2, o3, o4=inactive]            
            - orders = [5,2,2,1]
            - order_distribution = orders/sum(orders)
        - output:
            - n_orders describes the number of agent orders per price level
            - active_orders_within_levels is a set of order ids which are within the first self.n_levels levels
        """
        best_bid = self.get_best_price(side='bid')
        n_orders_within = []
        active_orders_within_levels = set()
        for l in range(0, self.n_levels):
            # this goes from 0 to self.n_levels-1 inclusive
            orders_on_level = 0
            if best_bid+l+1 in self.price_map['ask']:
                # TODO: copy is probably not necessary here 
                level = self.price_map['ask'][best_bid+l+1].copy()
                for order_id in level:
                    if self.order_map[order_id]['agent_id'] == 'agent':                        
                        orders_on_level += 1 
                        active_orders_within_levels.add(order_id)
                n_orders_within.append(orders_on_level)
            else:
                n_orders_within.append(0)
        # active remaining volume = volume which is still posted in the book, but not on the first three levels 
        assert self.volume == self.active_volume + self.inactive_volume
        # active remaining volume is the volume which is still posted in the book, but not on the first self.n_levels levels
        n_orders_outside = self.active_volume - sum(n_orders_within) 
        # add volume beyond the first three levels to the third level
        assert n_orders_outside >= 0
        assert len(n_orders_within) == self.n_levels
        assert n_orders_outside + sum(n_orders_within) == self.active_volume
        assert self.volume == self.active_volume + self.inactive_volume
        return n_orders_within, n_orders_outside, active_orders_within_levels
    
    def process_order_for_agent(self, order_type, side=None, price=None, order_id=None):
        """
        - process agent orders
        - modify active/inactive volume
        """
        if order_type == 'cancellation':
            order = {'type': 'cancellation', 'order_id': order_id, 'agent_id': 'agent'}
            self.process_order(order)
            self.inactive_volume += 1
            self.active_volume -= 1
        return None

    def _send_orders(self, action, additional_lots=0):
        """
        - find orders to be placed
        - example: current orders are [3,2,4,1], new action is [6,2,1,1], add orders to 2 first level, cancel 2 orders on the third level, leave other orders unchanged
        """

        ## cancel orders beyond the first self.n_levels 
        for order_id in self.active_orders.difference(self.order_info['order_ids_within_first_levels']):
            order = {'type': 'cancellation', 'order_id': order_id, 'agent_id': 'agent'}
            # TODO removal of order_id should be automatic within process order method 
            self.process_order(order)
            self.active_orders.remove(order_id)
            self.inactive_volume += 1
            self.active_volume -= 1 
        assert self.volume == self.active_volume + self.inactive_volume

        ##  
        target_n_orders = action_to_n_orders(action=action, volume=self.volume)
        # self.n_orders[-1] inactive orders, self.n_orders[-2] orders beyond first self.n_levels levels
        current_order_distribution = [0]+self.order_info['distribution_first_levels']+[self.inactive_volume]
        assert sum(current_order_distribution) == self.volume
        assert len(target_n_orders) == len(current_order_distribution)
        difference = [target_n_orders[n]-current_order_distribution[n] for n in range(len(current_order_distribution))]
        
        ## send limit orders and cancellations
        for n, level in zip(difference[1:-1], range(1, self.n_levels+1)):
            if n > 0:
                for _ in range(n):
                    # TODO: allow for limit order placement of size > 1
                    # TODO: 
                    order = {'agent_id': 'agent', 'type': 'limit', 'volume': 1, 'side': 'ask', 'price': self.get_best_price('bid')+level}
                    out = self.process_order(order)
                    self.active_orders.add(out[1])
                    # this is not needed, can do in terms of active orders 
                    self.active_volume += 1
                    self.inactive_volume -= 1
            elif n < 0:
                m = 0 
                price_level = self.price_map['ask'][self.get_best_price('bid')+level].copy()
                # cancel worst n orders on the price level 
                for order_id in price_level[::-1]:
                    if self.order_map[order_id]['agent_id'] == 'agent':
                        order = {'agent_id': 'agent', 'type': 'cancellation', 'volume': 1, 'order_id': order_id}
                        self.process_order(order)
                        self.active_orders.remove(order_id)
                        self.active_volume -= 1
                        self.inactive_volume += 1
                        m += 1
                    if m == -n:
                        break
            else:
                pass
        
        assert self.volume == self.active_volume + self.inactive_volume
        assert difference[0] == self.inactive_volume
        
        ## send market order
        if difference[0] > 0:
            order = {'agent_id': 'agent', 'type': 'market', 'volume': difference[0], 'side': 'bid'}
            out = self.process_order(order)
        else:
            out = None 

        return out  

    def step(self, action=None, transform_action=True):
        # warning: changed drift
        """
        the method returns observation, reward, terminated, truncated, info
        """
        assert len(action) == self.n_levels+2, "action must be of length n_levels + 2, additional 2 for market order and withholding volume"
        # transform action to allocation
        # TODO: softmax should be applied directly in the distribution class
        if transform_action:
            action = np.exp(action) / np.sum(np.exp(action), axis=0)
        else:
            pass

        ## check that all actions are positive 
        assert np.all(action >= 0)
        assert np.abs(np.sum(action) - 1) < 1e-3

        for order in self.active_orders:
            assert order in self.order_map
            
        reward = 0
        truncated = False
        terminated = False        

        # agent trades 
        # log shape of the book before trade

        if self.log:
            self.log_shape()
        out = self._send_orders(action=action)
        self.time += 1
        if out is not None:
            reward += self._get_reward(reward=out[2], traded_volume=out[3]['volume'])                                                
        if self.log:
            self.log_trade(out)
        assert self.volume >= 0
        assert self.volume == self.active_volume + self.inactive_volume
        if self.volume == 0:
            terminated = True
            return self._get_obs(), reward, terminated, truncated, {}
            
        # market trades
        for _ in range(99):
            if self.log:
               self.log_shape()

            out = self.generate_order() 
            self.time += 1
            if self.log:
                self.log_trade(out)

            assert self.time <= self.total_n_steps

            # cancel far out orders
            # self.cancel_far_out_orders()
            # check if limit orders of agent are filled
            if out[0] == 'market':
                if out[1]['agent']:
                    for order in out[1]['agent']: 
                        # note: only single size limit orders 
                        reward += self._get_reward(reward=order['price'], traded_volume=1)
                        self.active_orders.remove(order['order_id'])
                        self.volume -= 1
                        # TODO: this is not necessary, can do in terms of active orders
                        self.active_volume -= 1
                    assert self.volume >= 0  
                    assert self.active_volume >= 0
                    if self.volume == 0:
                        terminated = True         
                        return self._get_obs(), reward, terminated, truncated, {}
            
            for order in self.active_orders:
                assert order in self.order_map  

            # TODO: add drift later 
            # if self.drift_down:
            #     order = {'agent_id': self.market_agent_id, 'type': 'market', 'side': 'bid', 'volume': 1}
            #     self.process_order(order)
            # else:
            #     pass

            # handle terminal time 
            if self.time == self.total_n_steps:
                truncated = True
                terminated = True
                # remove active orders from the book 
                assert len(self.active_orders) == self.volume 
                for order in self.active_orders:
                    self.process_order({'type': 'cancellation', 'order_id': order, 'agent_id': 'agent'})
                    # TODO: remove active and inactive logic 
                    self.active_volume -= 1
                    self.inactive_volume += 1                    
                # send remaining volume as market order 
                assert self.active_volume == 0
                assert self.inactive_volume == self.volume
                order = {'agent_id': self.agent_id, 'type': 'market', 'volume': self.volume, 'side': 'bid'}
                out = self.process_order(order)
                reward += self._get_reward(out[2], self.volume)
                self.volume = self.inactive_volume = 0 
                return self._get_obs(), reward, terminated, truncated, {}
            
        # measure reward relative to best bid reward - traded_volume*best_bid
        return self._get_obs(), reward, terminated, truncated, {}
    
    def _book_shape(self,level=10):
        # warning!: this only works for 3 levels at the moment         
        # volume of first 3 levels on bid and ask side 
        # use initial shape to do this ! 
        reference = np.array([365, 1080, 1684], dtype=np.float32)
        bid_prices, bid_volumes = self.level2(side='bid', level=level)
        ask_prices, ask_volumes = self.level2(side='ask', level=level)
        #   
        bid_volumes = np.array(bid_volumes, dtype=np.float32)
        ask_volumes = np.array(ask_volumes, dtype=np.float32)
        bid_volumes /= reference
        ask_volumes /= reference     
        out =  np.concatenate((bid_volumes, ask_volumes), dtype=np.float32)
        out = np.clip(out, 0, 1)   
        return out

    def _get_obs(self):
        '''
        - the components are (time, agents remaining volume, price, imbalance, agents order allocations, shape of the book).
        - order allocations are [pi(1), ..., pi(N)], where pi(N) is the percentage of volume which was not placed yet, pi(n) is the allocation for levels n=1,...,N-1. 
        - current shape of the book is [v_1, ..., v_{N-1}] for bid and ask side. the current shape of the book is normalized by the initial shape of the book.
        - TODO: locations of orders in the book, this would replace the allocations. Not sure what the best way for this encoding is.         
        - Market drift: as a numerical features (strategic setting)
        - Note: all features are treated as continuous, not one hot encoded
        - all features are normalized to [0,1]
        - general question, can we make the asset more small tick, i.e. with smaller queue sizes 
        '''

        ### features 
        time = self.time/self.total_n_steps
        volume = self.volume/self.initial_volume
        mid_price = (self.get_best_price('ask') + self.get_best_price('bid'))/2
        price_drift = 100*(mid_price  - self.initial_mid_price)/self.initial_mid_price
        price_drift = min(1.0, price_drift)
        price_drift = max(-1.0, price_drift)
        bid_volume = self._get_best_volume(side='bid')
        ask_volume = self._get_best_volume(side='ask')
        imbalance = ask_volume/(bid_volume+ask_volume)
        first_part = np.array([time, volume, price_drift, imbalance], dtype=np.float32)
        # order information 
        n_orders_within, n_orders_outside, active_order_ids_within = self._get_agents_order_distribution()      
        assert self.volume == sum(n_orders_within) + n_orders_outside + self.inactive_volume
        assert active_order_ids_within <= self.active_orders
        allocation = np.array(n_orders_within+[n_orders_outside]+[self.inactive_volume], dtype=np.float32)
        assert sum(allocation) == self.volume
        if np.sum(allocation) == 0:
            pass
        else:
            allocation /= np.sum(allocation)  
        # find volumes 
        volumes = self._book_shape(level=3)        
        ### set attributed related to order information, TODO: could also move this into info 
        self.order_info = {'distribution_first_levels': n_orders_within, 'order_ids_within_first_levels': active_order_ids_within}
        assert self.volume == self.active_volume + self.inactive_volume
        return np.concatenate((first_part, allocation, volumes))
        # return (np.array([time], dtype=np.float32), self.volume, price_drift, np.array([imbalance], dtype=np.float32))

    def _get_best_volume(self, side):
        best_price = self.get_best_price(side=side)
        level = self.price_map[side][best_price]
        v = 0 
        for order_id in level:
            order = self.order_map[order_id]
            v += order['volume']
        return v 

    ## logging 
    def log_shape(self,order=None):
        # TODO: store bid and ask volumes/prices as an attribute 
        # We need them at any step to generated orders 
        # Could also modify each price level one at a time. Would be more efficient
        L = 30
        bid_prices, bid_volumes = self.level2(side='bid', level=L)
        ask_prices, ask_volumes = self.level2(side='ask', level=L)
        self.bid_volumes.append(bid_volumes)
        self.ask_volumes.append(ask_volumes)
        self.bid_prices.append(bid_prices)
        self.ask_prices.append(ask_prices)
        # best bid and ask prices 
        idx = np.nonzero(bid_volumes)[0][0]
        self.best_bid_prices.append(bid_prices[idx])
        self.best_bid_volumes.append(bid_volumes[idx])
        idx = np.nonzero(ask_volumes)[0][0]
        self.best_ask_prices.append(ask_prices[idx])
        self.best_ask_volumes.append(ask_volumes[idx])

        return None
        
    def log_trade(self, order):
        if order is None:
            self.trades.append(None)
            return None
        if order[0] == 'market':
            # assert order[1]['market_agent'], f'print order: {order}'
            self.trades.append((order[3]['side'], order[3]['volume']))
        else:
            self.trades.append(None)
        return None
    


if __name__ == '__main__': 
    config = {'total_n_steps': int(1e3), 'log': True, 'seed':0, 'initial_volume': 500, 'env_type': 'simple', 'ada':False}
    M = Market(config=config)
    print(f'initial volume is {config["initial_volume"]}')
    rewards = []
    for n in range(10):
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

    print(f'mean reward is {np.mean(rewards)}')
    print(f'max reward is {np.max(rewards)}')
    print(f'min reward is {np.min(rewards)}')
    





































