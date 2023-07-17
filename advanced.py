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

        # config is dict with entries {'total_n_steps', 'log', 'initial_level'}
        # initial_level places the limit order at best_ask + initial_level 
        # also determines the dimension of observation space and action space as [-1,0,1, ... ,initial_level]

        self.np_random = np.random.default_rng(config['seed'])

        ## order book related attributes
        self.order_map = {}
        self.price_map = {'bid': SortedDict(neg), 'ask': SortedDict()}
        self.market_agent_id = 'market_agent'
        self.agent_id = 'agent'

        # update_n increases every time the the order book is updated 
        # not necessarily the same as time 
        # there might be multiple updates at the same time (example: several orders get cancelled at the same time, several orders are filled by a market order) 
        # actions submitted by a single agent all happen at the same time 
        self.update_n = 0 

        # time updates after agent order is process or after market order is process 
        self.time = 0 

        ## market related attributes
        self.limit_intensities = np.array([0.2842, 0.5255, 0.2971, 0.2307, 0.0826, 0.0682, 0.0631, 0.0481, 0.0462, 0.0321, 0.0178, 0.0015, 0.0001])
        self.limit_intensities = np.pad(self.limit_intensities, (0,30-len(self.limit_intensities)), 'constant', constant_values=(0))
        self.cancel_intensities = 1e-3*np.array([0.8636, 0.4635, 0.1487, 0.1096, 0.0402, 0.0341, 0.0311, 0.0237, 0.0233, 0.0178, 0.0127, 0.0012, 0.0001])
        self.cancel_intensities = np.pad(self.cancel_intensities, (0,30-len(self.cancel_intensities)), 'constant', constant_values=(0))
        self.market_intesity = 0.1237
        
        # self.initial_shape = np.ones(30)*500

        shape = np.load('/u/weim/lob/stationary_shape.npz')
        self.initial_shape = np.mean([shape['bid'], shape['ask']], axis=0)
        self.initial_shape = np.rint(self.initial_shape).astype(int)

        # lognormal distribution parameters 
        self.market_volume_parameters = {'mean':4.00, 'sigma': 1.19} 
        self.limit_volume_parameters = {'mean':4.47, 'sigma': 0.83}
        self.cancel_volume_parameters = {'mean':4.48, 'sigma': 0.82}

        ## action and observation space settings 
        # observation space = [0,1,2]
        # action space = [-1,0,1,2]
        self.initial_level = config['initial_level']
        # time, level, shortfall, queue position  
        self.observation_space = Tuple( ( Discrete(11), Discrete(self.initial_level+2, start=-1), Discrete(7, start=-3), Box(0,np.inf), Box(0,1)))
        self.action_space = Discrete(self.initial_level+2, start=-1)


        ## logging 
        self.log = config['log']
        self.bid_volumes = []
        self.ask_volumes = []
        self.bid_prices = []
        self.ask_prices = []
        self.trades = []

        self.best_bid_prices = []
        self.best_ask_prices = []
        self.best_bid_volumes = []
        self.best_ask_volumes = []

        self.time = 0 

        self.total_n_steps = config['total_n_steps']


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

        return 'limit', order_id

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

        probability = np.array([self.market_intesity, self.market_intesity, limit_intensity, limit_intensity, bid_cancel_intensity, ask_cancel_intensity])
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # empty the order book and initialize 
        self.order_map = {}
        self.price_map = {'bid': SortedDict(neg), 'ask': SortedDict()}
        self.initialize_book()
        self.initial_ask = self.get_best_price(side='ask')

        # submit limit order to the second ask level in the book 
        best_ask = self.get_best_price(side='ask')
        order = {'agent_id':self.agent_id , 'type':'limit', 'side':'ask', 'price':best_ask+self.initial_level, 'volume':1} 
        out = self.process_order(order)
        self.active_order = out[1]
        # self.level = level

        # reset logging 
        self.bid_volumes = []
        self.ask_volumes = []
        self.bid_prices = []
        self.ask_prices = []
        self.trades = []

        self.best_bid_prices = []
        self.best_ask_prices = []
        self.best_bid_volumes = []
        self.best_ask_volumes = []

        # reset time 
        self.time = 0 
        time = self.time / self.total_n_steps

        return self._get_obs(time, self.initial_level), {}
    
    def _get_reward(self, reward):
        return reward - self.initial_ask

    def step(self, action=0):
        """
        the method returns observation, reward, terminated, truncated, info
        """
            
        reward = 0
        truncated = False
        terminated = False

        # action is market order, then cancel order and send market order  
        if action == -1:
            order = {'agent_id': self.agent_id, 'type': 'cancellation', 'order_id': self.active_order}                        
            self.process_order(order)
            order = {'agent_id': self.agent_id, 'type': 'market', 'side': 'bid', 'volume': 1}
            out = self.process_order(order)        
            time = self.time/self.total_n_steps
            reward = out[2]
            terminated = True     
            level = -1    
            self.active_order = None 
            return self._get_obs(time, level), self._get_reward(reward), terminated, truncated, {}
        
        else:
            # action is limit order, then leaver order or replace it 
            best_ask = self.get_best_price(side='ask')
            agent_order = self.order_map[self.active_order]
            current_level =  agent_order['price'] - best_ask        
            if action == current_level:
                pass            
            else:
                order = {'agent_id': self.agent_id, 'type': 'cancellation', 'order_id': self.active_order}                        
                self.process_order(order)
                order = {'agent_id': self.agent_id, 'type': 'limit', 'side': 'ask', 'price': best_ask+action, 'volume': 1}
                out = self.process_order(order)
                self.active_order = out[1]

        for _ in range(100):
            if self.log:
                self.logging()

            # transition to next state by generatating a random order 
            out = self.generate_order() 
            self.time += 1
            time = self.time/self.total_n_steps
            if self.log:
                self.log_trade(out)

            # assert self.time <= self.total_n_steps
            # cancel far out orders
            # self.cancel_far_out_orders()

            # check if limit order of agent is filled
            if out[0] == 'market':
                if out[1]['agent']:
                    self.active_order = None                
                    reward = out[1]['agent'][0]['price']
                    level = -1
                    terminated = True        
                    return self._get_obs(time, level), self._get_reward(reward), terminated, truncated, {}

            # handle terminal time 
            if self.time == self.total_n_steps:
                self.active_order = None
                truncated = terminated = True
                reward = self.get_best_price('bid') 
                level = -1
                return self._get_obs(time, level), self._get_reward(reward), terminated, truncated, {}
        
        # order not filled yet 
        # compute observation 
        agent_order = self.order_map[self.active_order]
        level = agent_order['price'] - self.get_best_price(side='ask')
        # adjust level if order drifts outside the boundary
        if level > self.initial_level:
            order = {'agent_id': self.agent_id, 'type': 'cancellation', 'order_id': self.active_order} 
            self.process_order(order)
            order = {'agent_id': self.agent_id, 'type': 'limit', 'side': 'ask', 'price': self.get_best_price(side='ask')+self.initial_level, 'volume': 1}
            out = self.process_order(order)
            self.active_order = out[1]
            level = self.initial_level
        else: 
            pass
        return self._get_obs(time, level), reward, terminated, truncated, {}
    
    def _get_obs(self, time, level):
        # include the current shortfall 
        shortfall = self.get_best_price('bid')  - self.initial_ask
        shortfall = max(-3, min(3, shortfall))
        # include queue position of order
        if self.active_order == None:
            queue_position = 0 
        else:
            queue_position = self.find_queue_position(self.active_order)/4000
        #
        time = self.total_n_steps*time  
        time = np.rint(time/100).astype(np.int32)
        # time = np.array([time], dtype=np.float32)
        queue_position = np.array([queue_position], dtype=np.float32)
        # include order book imbalance 
        bid_volume = self._get_best_volume(side='bid')
        ask_volume = self._get_best_volume(side='ask')
        imbalance = ask_volume/(bid_volume+ask_volume)
        imbalance = np.array([imbalance], dtype=np.float32)                
        return (time, level, shortfall, queue_position, imbalance)

    def _get_best_volume(self, side):
        best_price = self.get_best_price(side=side)
        level = self.price_map[side][best_price]
        v = 0 
        for order_id in level:
            order = self.order_map[order_id]
            v += order['volume']
        return v 

    ## logging 
    def logging(self,order=None):
        # TODO: store bid and ask volumes/prices as an attribute 
        # We need them at any step to generated orders 
        # Could also modify each price level one at a time. Would be more efficient
        L = 30
        bid_prices, bid_volumes = self.level2(side='bid', level=L)
        ask_prices, ask_volumes = self.level2(side='ask', level=L)
        self.bid_volumes.append(bid_volumes)
        self.ask_volumes.append(ask_volumes)
        # self.bid_prices.append(self.get_best_price(side='bid'))
        # self.ask_prices.append(self.get_best_price(side='ask'))
        # best bid and ask prices 
        idx = np.nonzero(bid_volumes)[0][0]
        self.best_bid_prices.append(bid_prices[idx])
        self.best_bid_volumes.append(bid_volumes[idx])
        idx = np.nonzero(ask_volumes)[0][0]
        self.best_ask_prices.append(ask_prices[idx])
        self.best_ask_volumes.append(ask_volumes[idx])

        return None
    
    def log_trade(self, order):
        if order[0] == 'market':
            # assert order[1]['market_agent'], f'print order: {order}'
            self.trades.append((order[3]['side'], order[3]['volume']))
        else:
            self.trades.append(None)
        return None
    

    ## plotting methods 

    # plot order book 
    def plot_level2_order_book(self, level=30, side='bid'):
        bid_prices, bid_volumes = self.level2(level=level, side='bid')
        ask_prices, ask_volumes = self.level2(level=level, side='ask')
        plt.figure()
        plt.bar(bid_prices, bid_volumes, color='b')
        plt.bar(ask_prices, ask_volumes, color='r')
    
    # plot average shape of order book 
    def plot_average_book_shape(self):
        """
        - plots average book shape.
        - if there is no config file, it saves the average book shape and other simulation parameters to a pickle file.
        """
        # book_shape['config'] = {'L': self.L, 'LR': self.LR, 'MR': self.MR, 'CR': self.CR}
        book_shape_bid = np.mean(self.bid_volumes, axis=0)
        book_shape_ask = np.mean(self.ask_volumes, axis=0)

        # np.savez('cont_model/stationary_shape', bid=book_shape_bid, ask=book_shape_ask)

        # _ = outfile.seek(0)

        # npzfile = np.load(outfile)

        # sorted(npzfile.files)
        # ['x', 'y']

        # npzfile['x']
        # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        plt.figure()
        plt.bar(range(0,-30,-1), book_shape_bid, color='red', label='bid')
        plt.bar(range(1,31,1), book_shape_ask, color='blue', label='ask')
        plt.legend(loc='upper right')
        plt.xlabel('relative distance to mid price')
        plt.ylabel('average volume')
    
    # plot prices 
    def plot_prices(self):
        """
        the method plots 
            - bid and ask prices 
            - microprice 
            - trades on bid and ask (larger trade with larger marker size)            
        """

        bid_prices = np.array(self.best_bid_prices)
        ask_prices = np.array(self.best_ask_prices)
        bid_volume = np.array(self.best_bid_volumes)
        ask_volume = np.array(self.best_ask_volumes)
        microprice = (bid_prices * ask_volume + ask_prices * bid_volume) / (bid_volume + ask_volume)
        time = np.arange(0, len(bid_prices))


        trades = [x[0] if x is not None else False for x in self.trades]
        bid_mask = [True if x == 'bid' else False for x in trades]
        ask_mask = [True if x == 'ask' else False for x in trades]

        plt.figure()
        plt.plot(time, bid_prices, '--', color='grey')
        plt.plot(time, ask_prices, '--', color='grey')
        plt.plot(time, microprice, '-', color='blue')

        plt.scatter(time[bid_mask], bid_prices[bid_mask], color='red', marker='x')
        plt.scatter(time[ask_mask], ask_prices[ask_mask], color='green', marker='x')
        

        #ToDo: log the market trades 

        return None 

## measure time 


if __name__ == '__main__': 
    # note: 
    # - logging or not makes difference in time
    # - cancellation of far out orders makes difference in time 
    import time 
    start = time.time()
    config = {'total_n_steps': int(1e3), 'log': True, 'seed':8, 'initial_level': 2}
    Market = Market(config=config)
    # assert (np.array([0.5], dtype=np.float32), -1) in Market.observation_space
    # assert (np.array([0.5], dtype=np.float32), Market.initial_level) in Market.observation_space
    # assert (np.array([0.5], dtype=np.float32), Market.initial_level+1) not in Market.observation_space
    rewards = []
    times = []
    shortfalls = []
    for n in range(2000):
        if n%100 == 0:
            print(f'episode {n}')
        Agent = SubmitAndLeaveAgent(level=0)
        observation, _ = Market.reset()
        observations = []
        actions = [] 
        q = []
        l = []
        terminated = truncated = False 
        while not terminated and not truncated: 
            # action = Agent.get_action(observation)
            action = 0
            # action = Market.action_space.sample()
            # action = 2
            assert action in Market.action_space
            active_order = Market.active_order
            observation, reward, terminated, truncated, info = Market.step(action)
            q.append(observation[3])
            l.append(observation[1])
            shortfalls.append(observation[2])
            times.append(Market.time)
            assert observation in Market.observation_space
            observations.append(observation)
        rewards.append(reward)

    elapsed = time.time()-start
    print(f'time elapsed in seconds: {elapsed}')

    # print(shortfalls)
    # print(times)
    plt.figure()
    Market.plot_prices()
    plt.savefig('prices.png')
    # plt.savefig('prices.png')
    # plt.plot(q)
    # plt.figure()
    # plt.plot(l)
    # plt.savefig('queue.png')

    print(f'average reward is {np.mean(rewards)}')
    print(f'max reward is {np.max(rewards)}')
    print(f'min reward is {np.min(rewards)}')
    print('done')

    rewards = np.array(rewards)
    np.save('rewards_bench', rewards)
    
    




































