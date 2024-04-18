import sys
import os 
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import numpy as np
from limit_order_book.limit_order_book import LimitOrder, MarketOrder, CancellationByPriceVolume, Cancellation
from config.config import config



class NoiseAgent(): 
    def __init__(self, rng, config_n,initial_shape_file=None, initial_shape=50, damping_factor=0.0, imbalance_reaction=False, imbalance_n_levels=4, level=30):
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
        if initial_shape_file is None:
            self.initial_shape = np.array([initial_shape]*self.initial_level)            
        else:
            shape = np.load(initial_shape_file) 
            self.initial_shape = np.clip(np.rint(np.mean([shape['bid_shape'], shape['ask_shape']], axis=0)), 1, np.inf)      
        # self.initial_shape = self.initial_level*[50]

        self.agent_id = 'noise_agent'

        return None  
    
    def reset_random_seet(self, rng):      
        self.np_random = rng
        return None
    
    def initialize(self, time): 
        # ToDo: initial bid and ask as variable 
        orders = [] 
        for idx, price in enumerate(np.arange(self.initial_bid, self.initial_bid-self.initial_level, -1)):
            order = LimitOrder(agent_id=self.agent_id, side='bid', price=price, volume=self.initial_shape[idx], time=time)
            orders.append(order)
        for idx, price in enumerate(np.arange(self.initial_ask, self.initial_ask+self.initial_level, 1)): 
            order = LimitOrder(agent_id=self.agent_id, side='ask', price=price, volume=self.initial_shape[idx], time=time)
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


    def sample_order(self, lob, time):
        """
        This methods samples an order and a time to execute this order 

        Parameters:
            - the current order book: it is expected to have the attributes best_bid_price, best_ask_price, bid_volumes, ask_volumes   
            - the current time 

        Returns:
            - order: class instance of LimitOrder, MarketOrder or CancellationByPriceVolume
            - time: float item time 

        """
        best_bid_price = lob.data.best_bid_prices[-1]
        best_ask_price = lob.data.best_ask_prices[-1]
        bid_volumes = lob.data.bid_volumes[-1]
        ask_volumes = lob.data.ask_volumes[-1]

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
        waiting_time = self.np_random.exponential(np.sum(probability))
        # waiting_time = 1 
        # time += waiting_time

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
            order = LimitOrder(agent_id=self.agent_id, side=side, price=price, volume=volume, time=time) 

        elif action == 'market':
            order = MarketOrder(agent_id=self.agent_id, side=side, volume=volume, time=time)
        
        elif action == 'cancellation':
            if side == 'bid':
                probability = self.cancel_intensities*bid_volumes/np.sum(self.cancel_intensities*bid_volumes)
                level = self.np_random.choice(np.arange(1, self.level+1), p=probability)       
                price = best_ask_price - level
            elif side == 'ask':
                probability = self.cancel_intensities*ask_volumes/np.sum(self.cancel_intensities*ask_volumes)
                level = self.np_random.choice(np.arange(1, self.level+1), p=probability)       
                price = best_bid_price + level
            order = CancellationByPriceVolume(agent_id=self.agent_id, side=side, price=price, volume=volume, time=time)
                            
        return order, waiting_time 

    def cancel_far_out_orders(self, lob, time):        
        # ToDo: Could add this as a function to the order book (as an order type)
        order_list = []
        for price in lob.price_map['bid'].keys():
            if price < lob.get_best_price('ask') - self.level:
                for order_id in lob.price_map['bid'][price]:
                    # this is repetitive 
                    if lob.order_map[order_id].agent_id == self.agent_id:
                        order = Cancellation(agent_id=self.agent_id, order_id=order_id, time=time)
                        order_list.append(order)
        # try to establish boundary conditions 
        # order = LimitOrder(agent_id=self.agent_id, side='bid', price=lob.get_best_price('ask') - self.level - 1, volume=self.initial_shape[0])
        # order_list.append(order)

        for price in lob.price_map['ask'].keys():
            if price > lob.get_best_price('bid') + self.level:
                for order_id in lob.price_map['ask'][price]:
                    # also repetitive 
                    if lob.order_map[order_id].agent_id == self.agent_id:
                        order = Cancellation(agent_id=self.agent_id, order_id=order_id, time=time)
                        order_list.append(order)                                
        # order = LimitOrder(agent_id=self.agent_id, side='ask', price=lob.get_best_price('bid') + self.level + 1, volume=self.initial_shape[0])
        # order_list.append(order)

        return order_list



class ExecutionAgent():
    """
    Base class for execution agents.
        - keeps track of volume, active_volume, cummulative_reward, passive_fills, market_fills
        - active volume is the volume currently placed in the book 
        - update positin takes a message and updates volumes, passive fille, market fills, and rewards  
        - reset function 
    """
    def __init__(self, volume, agent_id) -> None:
        self.initial_volume = volume
        self.frequency = 100
        self.agent_id = agent_id
        self.reset()
    
    def reset(self):
        self.volume = self.initial_volume
        self.active_volume = 0
        self.cummulative_reward = 0
        self.reference_bid_price = None
        self.market_buys = 0 
        self.market_sells = 0
        self.limit_buys = 0
        self.limit_sells = 0

    def get_reward(self, cash, volume):        
        assert self.reference_bid_price is not None, 'reference bid price is not set'
        return (cash - volume * self.reference_bid_price) / self.initial_volume
    
    def update_position_from_message_list(self, message_list):
        rewards = [self.update_position(m) for m in message_list]                
        assert self.volume >= 0         
        terminated = self.volume == 0
        return sum(rewards), terminated

    def update_position(self, fill_message):
        reward = 0 
        assert self.active_volume >= 0
        assert self.volume >= 0
        assert self.limit_buys >= 0
        assert self.limit_sells >= 0
        assert self.market_buys >= 0
        assert self.market_sells >= 0
        assert self.active_volume <= self.volume
        assert self.market_buys + self.market_sells + self.limit_buys + self.limit_sells <= self.initial_volume
        if fill_message.type == 'modification':
            # this agent doesnt modify orders
            print('MODIFICATION but we do not update the position!')
            pass
        elif fill_message.type == 'cancellation_by_price_volume':
            # Note: cancellation by price and volume could also return a list of cancellations and modifications
            if fill_message.order.agent_id == self.agent_id:
                self.active_volume -= fill_message.filled_volume
            else:
                pass
        elif fill_message.type == 'limit':
            if fill_message.agent_id == self.agent_id:
                # this means the limit order was send into the book by the agent 
                self.active_volume += fill_message.volume
        elif fill_message.type == 'cancellation':
            if fill_message.order.agent_id == self.agent_id:
                self.active_volume -= fill_message.volume
        elif fill_message.type == 'market':
            # check for active market trades             
            side = fill_message.order.side
            if fill_message.order.agent_id == self.agent_id:
                assert side == 'bid', 'this execution agent only sells'
                # ask means buy --> volume increases, negative cash flow 
                if side == 'ask':
                    self.volume += fill_message.filled_volume
                    self.market_buys += fill_message.filled_volume
                    reward -= self.get_reward(fill_message.execution_price, fill_message.filled_volume)
                    self.cummulative_reward -= reward
                # bid means sell --> volume decreases, positive cash flow
                elif side == 'bid':
                    self.volume -= fill_message.filled_volume
                    self.market_sells += fill_message.filled_volume
                    reward += self.get_reward(fill_message.execution_price, fill_message.filled_volume)
                    self.cummulative_reward += reward
            # check for passive limit order fills 
            if self.agent_id in fill_message.passive_fills:
                assert side == 'ask', 'this execution agent only sells'
                cash = 0
                volume = 0 
                for m in fill_message.passive_fills[self.agent_id]:
                    volume += m.filled_volume
                    cash += m.filled_volume * m.order.price 
                # market side is ask, market buy --> limit sell 
                if side == 'ask':
                    self.active_volume -= volume
                    self.volume -= volume
                    self.limit_sells += volume
                    reward += self.get_reward(cash, volume)
                    self.cummulative_reward += reward
                # market side is bid, market sell --> limit buy
                elif side == 'bid':
                    self.active_volume -= volume
                    self.volume += volume
                    self.limit_buys += volume
                    reward -= self.get_reward(cash, volume)
                    self.cummulative_reward -= reward
            return reward
        else: 
            raise ValueError(f'Unknown message type {fill_message.type}')
        return 0
    
    def sell_remaining_position(self, lob, time):
        assert self.agent_id, 'agent id is not set' 
        assert lob.order_map_by_agent[self.agent_id], 'agent has no orders in the book'
        # assert self.agent_id in lob.order_map_by_agent, 'agent has no orders in the book'
        order_list = []
        for order_id in lob.order_map_by_agent[self.agent_id]:
            order_list.append(Cancellation(agent_id=self.agent_id, order_id=order_id, time=time))
        order_list.append(MarketOrder(agent_id=self.agent_id, side='bid', volume=self.volume, time=time))
        return order_list

class MarketAgent(ExecutionAgent):

    def __init__(self, volume, agent_id) -> None:
        super().__init__(volume, 'market_agent')
        self.when_to_place = 0 
                    
    def generate_order(self, time, lob):
        assert self.active_volume >= 0 
        assert self.volume >= 0
        if time == self.when_to_place:
            self.reference_bid_price = lob.get_best_price('bid')
            return [MarketOrder(self.agent_id, side='bid', volume=self.volume, time=time)]
        else:
            return None
    
    def get_observation(self, time, lob):
        return None 

class SubmitAndLeaveAgent(ExecutionAgent):

    def __init__(self, volume, terminal_time=100) -> None:
        super().__init__(volume, 'sl_agent')
        self.terminal_time = terminal_time
        self.when_to_place = 0 
                        
    def generate_order(self, time, lob):
        assert self.volume >= 0
        assert self.active_volume >= 0 
        assert time <= self.terminal_time        
        if time == self.when_to_place:
            self.reference_bid_price = lob.get_best_price('bid')
            limit_price = lob.get_best_price('bid')+1
            # print(f'reference bid price is {self.reference_bid_price}')
            # print(f'placing limit order at {limit_price}')            
            return [LimitOrder(self.agent_id, side='ask', price=limit_price, volume=self.initial_volume, time=time)]
        else:
            return None
    
    def get_observation(self, time, lob):
        return None 

class LinearSubmitLeaveAgent(ExecutionAgent):

    def __init__(self, volume, terminal_time=1000, frequency=100) -> None:
        super().__init__(volume)
        self.agent_id = 'linear_sl_agent'
        self.terminal_time = terminal_time
        self.when_to_place = 0 
        self.frequency = frequency
        steps = int(terminal_time/frequency)
        assert volume % steps == 0 or volume < steps
        assert self.when_to_place < self.terminal_time
        if volume >= steps:
            self.volume_slice = int(self.initial_volume/steps)
            assert self.volume_slice * steps == volume
        return None 
                        
    def generate_order(self, time, lob):
        assert self.volume >= 0
        assert self.active_volume >= 0 
        if time == self.when_to_place:
            self.reference_bid_price = lob.get_best_price('bid')
        if time % self.frequency == 0 and time < self.terminal_time:
            limit_price = lob.get_best_price('bid')+1
            return [LimitOrder(self.agent_id, side='ask', price=limit_price, volume=self.volume_slice, time=time)]
        else:
            return None
    
    def get_observation(self, time, lob):
        return None 

class RLAgent(ExecutionAgent):
    """
        - this agent takes in an action and then generates an order
    """
    def __init__(self, volume, terminal_time) -> None:
        super().__init__(volume, 'rl_agent')
        self.orders_within_range = set()
        self.when_to_place = 0
        self.terminal_time = terminal_time
        
    
    def generate_order(self, time, lob, action):
        """
        - generate list of orders from an action
        - return the list of orders
        """
        if time == self.when_to_place:
            self.reference_bid_price = lob.get_best_price('bid') 

        best_bid = lob.get_best_price('bid')

        action = np.exp(action) / np.sum(np.exp(action), axis=0)
        # print(action.round(2))

        
        # keep track of total cancelled volume 
        cancelled_volume = 0
        order_list = []
        # self.orders_within_range is set in the get_observation function
        # cancel orders that are not on levels >= l4 
        orders_to_cancel = lob.order_map_by_agent[self.agent_id].difference(self.orders_within_range)
        for order_id in orders_to_cancel:
            order_list.append(Cancellation(self.agent_id, order_id))
            cancelled_volume += lob.order_map[order_id].volume
        
        assert cancelled_volume == self.volume_per_level[-1]
        
        # target volumes 
        target_volumes = []
        available_volume = self.volume 
        for l in range(len(action)):
            # [market,l1 ,l2, ,l3, inactive]
            # np.round rounds values in [0, 0.5] to 0, and values in [0.5, 1] to 1 
            volume_on_level = min(np.round(action[l]*self.volume).astype(int), available_volume)
            available_volume -= volume_on_level
            target_volumes.append(volume_on_level) 
        target_volumes[-1] += available_volume
        
        # current volumes contains levels l1, l2, l3, >=l4
        current_volumes = self.volume_per_level
        current_volumes.extend([self.volume - self.active_volume + cancelled_volume]) 
        current_volumes.insert(0, 0)

        l = 0 
        m = 0 
        # market_sells = []
        for level in range(4): 
            # 0, 1, 2, 3
            if level == 0:
                if target_volumes[level] > 0:
                    order = MarketOrder(self.agent_id, 'bid', target_volumes[level])
                    order_list.append(order)
                    # market_sells.append(order)
                    m = target_volumes[level]
            else:
                diff = target_volumes[level] - current_volumes[level]
                limit_price = best_bid+level
                if diff > 0:
                    order_list.append(LimitOrder(self.agent_id, 'ask', limit_price, diff))
                    l += diff
                elif diff < 0:
                    order_list.insert(0, CancellationByPriceVolume(agent_id=self.agent_id, side='ask', price=limit_price, volume=-diff))
                    cancelled_volume += -diff
                else:
                    pass

        # new limit and market order must match the cancelled volume
        assert l >= 0
        assert m >= 0
        assert self.volume >= self.active_volume >= cancelled_volume >= 0
        free_budget = cancelled_volume + (self.volume - self.active_volume)
        assert 0 <= free_budget <= self.volume
        assert 0 <= l + m <= free_budget, 'l + m must be less than or equal to the free budget'
        # target volumes[-1] = inactive = (volume - active volume) + cancelled - l - m 
        assert target_volumes[-1] == (self.volume - self.active_volume) + cancelled_volume - l - m 
        # TODO: create check to test validity of the order list
        return order_list

    
    def get_observation(self, time, lob):        
        best_bid = lob.get_best_price(side='bid')
        volume_per_level = []
        orders_within_range = set()
        #TODO: remove the hard coding of levels here 
        for level in range(1, 4):
            # 1,2,3
            orders_on_level = 0
            if best_bid+level in lob.price_map['ask']:
                for order_id in lob.price_map['ask'][best_bid+level]:
                    if lob.order_map[order_id].agent_id == self.agent_id:                        
                        orders_on_level += lob.order_map[order_id].volume
                        orders_within_range.add(order_id)
                volume_per_level.append(orders_on_level)
            else:
                volume_per_level.append(0)
        # append volumes on levels >=4
        volume_per_level.append(self.active_volume-sum(volume_per_level))
        assert sum(volume_per_level) == self.active_volume
        # append volumes that are inactive 
        # volume_per_level.append(self.volume-self.active_volume)
        assert sum(volume_per_level) == self.active_volume
        self.orders_within_range = orders_within_range
        self.volume_per_level = volume_per_level
        assert sum(self.volume_per_level) <= self.volume
        # 
        if time == 0:
            bid_move = 0
        else:
            bid_move = (best_bid - self.reference_bid_price)/10
        spread = (lob.get_best_price('ask') - best_bid)/10
        _, bid_v = lob.level2('bid')
        _, ask_v = lob.level2('ask')

        # probably a good idea to make this unit less. 
        # should also take imbalances     
        bid_v = bid_v[:3]/np.array([5,10,20])
        ask_v = ask_v[:3]/np.array([5,10,20])   

        volume_per_level = np.array(volume_per_level, dtype=np.float32)/self.initial_volume

        out = np.array([time/self.terminal_time, self.volume/self.initial_volume, self.active_volume/self.initial_volume, (self.volume-self.active_volume)/self.initial_volume, bid_move, spread], dtype=np.float32)

        out = np.concatenate([out, volume_per_level, bid_v, ask_v], dtype=np.float32)

        # add 4th level imbalance
        if np.sum(bid_v[:6]) + np.sum(ask_v[:6]) == 0:
            # print('zero bid/ask volumes')
            # print(bid_v[:4])
            # print(ask_v[:4])            
            imbalance = 0
        else:
            # print('zero bid/ask volumes')
            # print(bid_v[:4])
            # print(ask_v[:4])       
            imbalance = (np.sum(bid_v[:6]) - np.sum(ask_v[:6]))/(np.sum(bid_v[:6]) + np.sum(ask_v[:6]))
        imbalance = np.array([imbalance], dtype=np.float32)
        # out = np.append(out, imbalance, dtype=np.float32)
        out = np.concatenate([out, imbalance])
        # print(f'imbalance: {imbalance}')

        # 
        return out 
            
class StrategicAgent():
    """
    - just sends limit and market orders at some frequency
    - we do not keep track of the agents position 
    """
    def __init__(self, frequency, market_volume, limit_volume, rng, offset) -> None:
        assert 0 <= offset < frequency, 'offset must be in {0,1, ..., frequency-1}'        
        self.frequency = frequency
        self.offset = offset 
        self.market_order_volume = market_volume
        self.limit_order_volume = limit_volume
        self.agent_id = 'strategic_agent'
        self.rng = rng
        self.direction = None 
        return None 
    
    def generate_order(self, lob, time):        
        if self.direction == 'sell':                
            limit_price = lob.get_best_price('bid')+1
            order_list = []
            order_list.append(MarketOrder(self.agent_id, 'bid', self.market_order_volume, time))
            order_list.append(LimitOrder(self.agent_id, 'ask', limit_price, self.limit_order_volume, time))                        
            return order_list
        elif self.direction == 'buy':
            limit_price = lob.get_best_price('ask')-1
            order_list = []
            order_list.append(MarketOrder(self.agent_id, 'ask', self.market_order_volume, time))
            order_list.append(LimitOrder(self.agent_id, 'bid', limit_price, self.limit_order_volume, time))                        
            return order_list
        else:
            raise ValueError(f'direction must be either buy or sell, got {self.direction}')

        
    def reset(self):
        self.direction = self.rng.choice(['buy', 'sell'])
        return None
