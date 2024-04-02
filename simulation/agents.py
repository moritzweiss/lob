import sys
import os 
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import numpy as np
from limit_order_book.limit_order_book import LimitOrder, MarketOrder, CancellationByPriceVolume, Cancellation
from config.config import config


class NoiseAgent(): 
    def __init__(self, rng, config_n, initial_shape_file, damping_factor=0.0, imbalance_reaction=False, imbalance_n_levels=4, level=30):
        """"    
        Parameters:
        ----
        initial_shape_file: str
            stationary shape of the order book to start from 
        level: int 
            number of levels the noise agents placed its orders. also the number of levels to initialize the order book.
            numbers of orders outside this range are cancelled
        imbalance_reaction: bool 
            if True, the agent reacts to the imbalance of the order book 
        config_n: int 
            number of the configuration to use, 0: is from paper, 1: is config for small tick stocks 
            the configs contain intensities for limit orders, market orders and cancellations, as well as volume parameters
        number of levels to compute the imbalance: int 
            number of levels to compute the imbalance. if this is =1, imbalance is just computed from top of the book
            the parameters does not have any effect if imbalance_reaction is False  
        rng: np.random number generator instance                                 
        """

        self.damping_factor = damping_factor        
        self.damping_weights = np.exp(-self.damping_factor*np.arange(level)) # move damping weights here for speed up
        
        self.imbalance_n_levels = imbalance_n_levels # anything related to this is commented out at the moment  

        self.config = config[config_n]

        self.imbalance_reaction = imbalance_reaction

        self.np_random = rng 
        self.level = level 
        self.initial_level = self.level # number of levels to initialize (could be bigger than self.level)
        assert self.initial_level >= self.level, 'initial level must be bigger than level'
        self.initial_bid = 1000
        self.initial_ask = 1001
                
        # load intensities 
        limit_intensities = self.config['intensities']['limit']
        market_intensity = self.config['intensities']['market']
        cancel_intensities = self.config['intensities']['cancellation']

        assert self.level >= 10, 'level must be at least 10'
        self.limit_intensities = np.pad(limit_intensities, (0,self.level-len(limit_intensities)), 'constant', constant_values=(0))
        self.cancel_intensities = np.pad(cancel_intensities, (0,self.level-len(cancel_intensities)), 'constant', constant_values=(0))
        self.market_intesity = market_intensity

        self.distribution = self.config['distribution']
        self.market_volume_parameters = self.config['volumes']['market']
        self.limit_volume_parameters = self.config['volumes']['limit']
        self.cancel_volume_parameters = self.config['volumes']['cancellation']
        self.volume_min = self.config['volumes']['clipping']['min']
        self.volume_max = self.config['volumes']['clipping']['max']


        # initial shape of the order book
        shape = np.load(initial_shape_file) 
        self.initial_shape = np.clip(np.rint(np.mean([shape['bid_shape'], shape['ask_shape']], axis=0)), 1, np.inf)      
        # self.initial_shape = self.initial_level*[50]

        self.agent_id = 'noise_agent'

        return None  
    
    def reset_random_seet(self, rng):      
        self.np_random = rng
        return None
    
    def initialize(self): 
        # ToDo: initial bid and ask as variable 
        orders = [] 
        for idx, price in enumerate(np.arange(self.initial_bid, self.initial_bid-self.initial_level, -1)):
            order = LimitOrder(agent_id=self.agent_id, side='bid', price=price, volume=self.initial_shape[idx])
            orders.append(order)
        for idx, price in enumerate(np.arange(self.initial_ask, self.initial_ask+self.initial_level, 1)): 
            order = LimitOrder(agent_id=self.agent_id, side='ask', price=price, volume=self.initial_shape[idx])
            orders.append(order)
        return orders

    def volume(self, action):
        assert self.config['distribution'] in ['log_normal', 'half_normal_plus1'], 'distribution not implemented'
        # ToDo: initialize distribution at the beginning 
        if self.distribution == 'log_normal':
            if action == 'limit':
                volume = self.np_random.lognormal(self.limit_volume_parameters['mean'], self.limit_volume_parameters['std'])   
            elif action == 'market':
                volume = self.np_random.lognormal(self.market_volume_parameters['mean'], self.market_volume_parameters['std'])
            elif action == 'cancellation':
                volume = self.np_random.lognormal(self.cancel_volume_parameters['mean'], self.cancel_volume_parameters['std'])
            volume = np.rint(np.clip(1+np.abs(volume), self.volume_min, self.volume_max))

        else: 
            if action == 'limit':
                volume = self.np_random.normal(self.limit_volume_parameters['mean'], self.limit_volume_parameters['std'])   
            elif action == 'market':
                volume = self.np_random.normal(self.market_volume_parameters['mean'], self.market_volume_parameters['std'])
            elif action == 'cancellation':
                volume = self.np_random.normal(self.cancel_volume_parameters['mean'], self.cancel_volume_parameters['std'])
            volume = np.rint(np.clip(1+np.abs(volume), self.volume_min, self.volume_max))

        return volume


    def sample_order(self, best_bid_price, best_ask_price, bid_volumes, ask_volumes):
        ''''
        - input: shape of the limit order book 
        - output: order 
        '''

        # handling of nan best bid price 
        if np.isnan(best_bid_price):
            if np.isnan(best_ask_price):
                # bid and ask are nan 
                order = LimitOrder(agent_id=self.agent_id, side='bid', price=self.initial_bid, volume=self.initial_shape[0])
            else:
                # bid is nan, ask is not nan
                order = LimitOrder(agent_id=self.agent_id, side='bid', price=best_ask_price-1, volume=self.initial_shape[0])
            return order
        if np.isnan(best_ask_price):
            # bid is not nan, ask is nan
            order = LimitOrder(agent_id=self.agent_id, side='ask', price=best_bid_price+1, volume=self.initial_shape[0])
            return order

        assert len(bid_volumes) == len(ask_volumes), 'bid and ask volumes must have the same length'
        assert np.all(bid_volumes >= 0), 'All entries of bid volumes must be >= 0'
        assert np.all(ask_volumes >= 0), 'All entries of ask volumes must be >= 0'

        # n_levels = len(bid_volumes)
        

        ask_cancel_intensity = np.sum(self.cancel_intensities*ask_volumes)
        bid_cancel_intensity = np.sum(self.cancel_intensities*bid_volumes)
        limit_intensity = np.sum(self.limit_intensities)


        # check if bid and ask volumes are zero at the same time 
        if self.imbalance_reaction:
            # if (np.sum(bid_volumes) == 0) and (np.sum(ask_volumes) == 0):
            #     imbalance = 0
            # else:
                # imbalance = ((bid_volumes[0]) - ask_volumes[0])/(bid_volumes[0] + ask_volumes[0])
                # imbalance = (np.sum(bid_volumes[:self.imbalance_n_levels]) - np.sum(ask_volumes[:self.imbalance_n_levels]))/(np.sum(bid_volumes[:self.imbalance_n_levels]) + np.sum(ask_volumes[:self.imbalance_n_levels]))    
                # Compute imbalance with exponential damping
            # weights = np.exp(-np.arange(self.imbalance_n_levels))
            # c = 0.5
            # c = 1.0
            # c = 0.0  
            # weights = np.exp(-self.damping_factor*np.arange(len(bid_volumes)))
            weighted_bid_volumes = np.sum(self.damping_weights * bid_volumes)
            weighted_ask_volumes = np.sum(self.damping_weights * ask_volumes)
            if (weighted_bid_volumes + weighted_ask_volumes) == 0:
                imbalance = 0
            else:
                imbalance = (weighted_bid_volumes - weighted_ask_volumes) / (weighted_bid_volumes + weighted_ask_volumes)
            if np.isnan(imbalance):
                print(imbalance)
                print(bid_volumes)
                print(ask_volumes)
                raise ValueError('imbalance is nan')
            imbalance = np.sign(imbalance)*np.power(np.abs(imbalance), 1/2)
            # imbalance = np.sign(imbalance)
            market_buy_intensity = self.market_intesity*(1+imbalance)
            market_sell_intensity = self.market_intesity*(1-imbalance)
        else:
            market_buy_intensity = self.market_intesity
            market_sell_intensity = self.market_intesity


        probability = np.array([market_sell_intensity, market_buy_intensity, limit_intensity, limit_intensity, bid_cancel_intensity, ask_cancel_intensity])
        probability = probability/np.sum(probability)

        action, side = self.np_random.choice([('market', 'bid'), ('market', 'ask'), ('limit', 'bid'), ('limit', 'ask'), ('cancellation', 'bid'), ('cancellation', 'ask')], p=probability)


        volume = self.volume(action)

        if action == 'limit': 
            probability = self.limit_intensities/np.sum(self.limit_intensities)
            level = self.np_random.choice(np.arange(1, self.level+1), p=probability)       
            if side == 'bid': 
                price = best_ask_price - level
            else: 
                price = best_bid_price + level
            order = LimitOrder(agent_id=self.agent_id, side=side, price=price, volume=volume) 

        elif action == 'market':
            order = MarketOrder(agent_id=self.agent_id, side=side, volume=volume)
        
        elif action == 'cancellation':
            if side == 'bid':
                probability = self.cancel_intensities*bid_volumes/np.sum(self.cancel_intensities*bid_volumes)
                level = self.np_random.choice(np.arange(1, self.level+1), p=probability)       
                price = best_ask_price - level
            elif side == 'ask':
                probability = self.cancel_intensities*ask_volumes/np.sum(self.cancel_intensities*ask_volumes)
                level = self.np_random.choice(np.arange(1, self.level+1), p=probability)       
                price = best_bid_price + level
            order = CancellationByPriceVolume(agent_id=self.agent_id, side=side, price=price, volume=volume)
            
        return order 

    def cancel_far_out_orders(self, lob):        
        # ToDo: Could add this as a function to the order book (as an order type)
        order_list = []
        for price in lob.price_map['bid'].keys():
            if price < lob.get_best_price('ask') - self.level:
                for order_id in lob.price_map['bid'][price]:
                    # this is repetitive 
                    if lob.order_map[order_id].agent_id == self.agent_id:
                        order = Cancellation(agent_id=self.agent_id, order_id=order_id)
                        order_list.append(order)
        # try to establish boundary conditions 
        # order = LimitOrder(agent_id=self.agent_id, side='bid', price=lob.get_best_price('ask') - self.level - 1, volume=self.initial_shape[0])
        # order_list.append(order)

        for price in lob.price_map['ask'].keys():
            if price > lob.get_best_price('bid') + self.level:
                for order_id in lob.price_map['ask'][price]:
                    # also repetitive 
                    if lob.order_map[order_id].agent_id == self.agent_id:
                        order = Cancellation(agent_id=self.agent_id, order_id=order_id)
                        order_list.append(order)                                
        # order = LimitOrder(agent_id=self.agent_id, side='ask', price=lob.get_best_price('bid') + self.level + 1, volume=self.initial_shape[0])
        # order_list.append(order)

        return order_list
