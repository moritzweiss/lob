from os import environ
import os 
import sys 
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)
# N_THREADS = '1'
# environ['OMP_NUM_THREADS'] = N_THREADS
# environ['OPENBLAS_NUM_THREADS'] = N_THREADS
# environ['MKL_NUM_THREADS'] = N_THREADS
# environ['VECLIB_MAXIMUM_THREADS'] = N_THREADS
# environ['NUMEXPR_NUM_THREADS'] = N_THREADS
from typing import Any
from agents import NoiseAgent
import gymnasium as gym 
from limit_order_book.limit_order_book import LimitOrderBook, MarketOrder, LimitOrder, Cancellation, CancellationByPriceVolume
from limit_order_book.plotting import heat_map
import matplotlib.pyplot as plt
import numpy as np
import warnings
from gymnasium.spaces import Box
import os

# TODO: test benchmark strategies. Test that rewards are as expected. Can just hand code an environemnt for this. 

class DummyAgent():
  """
  TODO: can implement a dummy agent that does nothing for easier simulation  
  """

class ExecutionAgent():
    """
    Base class for execution agents.
        - keeps track of volume, active_volume, cummulative_reward, passive_fills, market_fills
        - active volume is the volume currently placed in the book 
        - update positin takes a message and updates volumes, passive fille, market fills, and rewards  
        - reset function 
    """
    def __init__(self, volume) -> None:
        self.initial_volume = volume
    
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
        return sum(rewards)

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

class MarketAgent(ExecutionAgent):

    def __init__(self, volume) -> None:
        super().__init__(volume)
        self.agent_id = 'market_agent'
        self.when_to_place = 0 
                    
    def generate_order(self, time, lob):
        if time == self.when_to_place:
            self.reference_bid_price = lob.get_best_price('bid')
            return MarketOrder(self.agent_id, side='bid', volume=self.volume)
        else:
            return None
    
    def get_observation(self, time, lob):
        return None 

class SubmitAndLeaveAgent(ExecutionAgent):

    def __init__(self, volume, terminal_time=100) -> None:
        super().__init__(volume)
        self.agent_id = 'sl_agent'
        self.terminal_time = terminal_time
        self.when_to_place = 0 
                        
    def generate_order(self, time, lob):
        assert time <= self.terminal_time        
        if time == self.when_to_place:
            self.reference_bid_price = lob.get_best_price('bid')
            limit_price = lob.get_best_price('bid')+1
            return LimitOrder(self.agent_id, side='ask', price=limit_price, volume=self.initial_volume)
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
        if time == self.when_to_place:
            self.reference_bid_price = lob.get_best_price('bid')
        if time % self.frequency == 0 and time < self.terminal_time:
            limit_price = lob.get_best_price('bid')+1
            return LimitOrder(self.agent_id, side='ask', price=limit_price, volume=self.volume_slice)
        else:
            return None
    
    def get_observation(self, time, lob):
        return None 

class RLAgent(ExecutionAgent):
    """
        - this agent takes in an action and then generates an order
    """
    def __init__(self, volume, terminal_time) -> None:
        super().__init__(volume)
        self.agent_id = 'rl_agent'  
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

        order_list = []
        
        cancelled_volume = 0
        order_list = []
        # self.orders_within_range is set in the get_observation function
        orders_to_cancel = lob.order_map_by_agent[self.agent_id].difference(self.orders_within_range)
        for order_id in orders_to_cancel:
            order_list.append(Cancellation(self.agent_id, order_id))
            cancelled_volume += lob.order_map[order_id].volume
        
        # target volumes 
        target_volumes = []
        available_volume = self.volume 
        for l in range(len(action)):
            # [market,1,2,3, inactive]
            # np.round rounds values in [0, 0.5] to 0, and values in [0.5, 1] to 1 
            volume_on_level = min(np.round(action[l]*self.volume).astype(int), available_volume)
            available_volume -= volume_on_level
            target_volumes.append(volume_on_level) 
        target_volumes[-1] += available_volume

        # generate orders
        current_volumes = self.volume_per_level
        # extend the inactive entry by volume-active_volume+cancelled_volume: entries are now [l1, l2, l3, inactive]
        current_volumes.extend([self.volume - self.active_volume + cancelled_volume]) 
        current_volumes.insert(0, 0)

        cancellations = []
        limit_orders = []
        # order_list = []
        c = 0 
        l = 0 
        m = 0 
        for level in range(4): 
            if level == 0:
                if target_volumes[level] > 0:
                    order_list.append(MarketOrder(self.agent_id, 'bid', target_volumes[level]))
                    m = target_volumes[level]
            else:
                diff = target_volumes[level] - current_volumes[level]
                limit_price = best_bid+level
                if diff > 0:
                    order_list.append(LimitOrder(self.agent_id, 'ask', limit_price, diff))
                    l += diff
                elif diff < 0:
                    order_list.insert(0, CancellationByPriceVolume(agent_id=self.agent_id, side='ask', price=limit_price, volume=-diff))
                    c += -diff
                else:
                    pass
        
        assert l + m - c <= self.volume
        assert target_volumes[-1] == (self.volume - self.active_volume) + cancelled_volume - l - m + c

        return order_list 

        # generate orders, need the current volume on each level
        # need current order distribution 
    
    def get_observation(self, time, lob):        
        best_bid = lob.get_best_price(side='bid')
        volume_per_level = []
        orders_within_range = set()
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
        self.orders_within_range = orders_within_range
        self.volume_per_level = volume_per_level
        assert sum(self.volume_per_level) <= self.volume
        return np.array([time/self.terminal_time, self.volume/self.initial_volume], dtype=np.float32)
            
class StrategicAgent():
    """
    - just sends limit and market orders at some frequency
    - we do not keep track of the agents position 
    """
    def __init__(self, frequency, market_volume, limit_volume, rng) -> None:
        self.frequency = frequency
        self.market_order_volume = market_volume
        self.limit_order_volume = limit_volume
        self.agent_id = 'strategic_agent'
        self.rng = rng
        self.direction = None 
        return None 
    
    def generate_order(self, time, lob):        
        if time % self.frequency == 0: 
            if self.direction == 'sell':                
                limit_price = lob.get_best_price('bid')+1
                order_list = []
                order_list.append(MarketOrder(self.agent_id, 'bid', self.market_order_volume))
                order_list.append(LimitOrder(self.agent_id, 'ask', limit_price, self.limit_order_volume))                        
                return order_list
            elif self.direction == 'buy':
                limit_price = lob.get_best_price('ask')-1
                order_list = []
                order_list.append(MarketOrder(self.agent_id, 'ask', self.market_order_volume))
                order_list.append(LimitOrder(self.agent_id, 'bid', limit_price, self.limit_order_volume))                        
                return order_list
            else:
                raise ValueError(f'direction must be either buy or sell, got {self.direction}')
        else: 
            return None 
        
    def reset(self):
        self.direction = self.rng.choice(['buy', 'sell'])
        return None

class Market(gym.Env):
    def __init__(self, config):                
                #  seed, type='noise', execution_agent='market', terminal_time=int(1e3), volume=20, level=30, damping_factor=0.5, market_volume=2, limit_volume=5, frequency=50) -> None:
        """ 
            - seed: controls seeding for the noise agent. other agents are deterministic 
            - type: noise, flow, strategic controls the type of market environment
            - benchmark_agent: None, market, linear_sl, sl, controls the type of benchmark agent (set to None without benchmark)
            - terminal_time: the time at which the environment terminates
            - market, limit volume and frequency are parameters for the strategic agent
            - damping factor is the exponential damping factor for the noise agent 
            - volume: how much volume the agent trades 

            TODO : 
            - damping factor should move into a config file once it is fixed 
            - level and terminal time should be moved into a config file 
            - market, limit, frequency, should be moved into a config file 
            - we should have one config option. like config=1 sets the options for all the other environments 
            
        """ 
        # initialize execution agent 
        if config['execution_agent'] == 'market_agent':
            self.execution_agent = MarketAgent(volume=config['volume'])
        elif config['execution_agent'] == 'sl_agent':
            self.execution_agent = SubmitAndLeaveAgent(volume=config['volume'], terminal_time=config['terminal_time'])
        elif config['execution_agent'] == 'linear_sl_agent':
            self.execution_agent = LinearSubmitLeaveAgent(volume=config['volume'], terminal_time=config['terminal_time'], frequency=100)
        elif config['execution_agent'] == 'rl_agent':
            self.execution_agent = RLAgent(volume=config['volume'], terminal_time=config['terminal_time'])
        elif config['execution_agent'] == None:
            self.execution_agent = None   
        else:
            raise ValueError(f"Unknown value for execution agent {config['execution_agent']}")
        
        # setting market environment 
        assert config['type'] in ['noise', 'flow', 'strategic'], f'Unknown type {config["type"]}'

        # setting noise agent
        if config['type'] == 'noise':
            imbalance = False
        else:
            imbalance = True
        self.noise_agent = NoiseAgent(level=config['level'], rng=np.random.default_rng(config['seed']), imbalance_reaction=imbalance, initial_shape_file=f'{parent_dir}/data_small_queue.npz', config_n=1, damping_factor=0.5)
        
        # setting strategic agent
        if config['type'] == 'strategic':
            self.strategic_agent = StrategicAgent(frequency=50, market_volume=2, limit_volume=5, rng=np.random.default_rng(config['seed']))
        else:
            self.strategic_agent = None

        # terminal time
        self.terminal_time = config['terminal_time']

        # observation space is time and inventory 
        self.observation_space = Box(low=-1, high=1, shape=(2,), seed=config['seed'], dtype=np.float32) 

        # action space [m, l1, l2, l3, inactive]
        self.action_space = Box(low=-10, high=10, shape=(5,), seed=config['seed'], dtype=np.float32)

        #
        super().reset(seed=config['seed'])


    def transition(self):
        reward = 0
        # noise trader trades at every time step        
        order = self.noise_agent.sample_order(self.lob.data.best_bid_prices[-1], self.lob.data.best_ask_prices[-1], self.lob.data.bid_volumes[-1], self.lob.data.ask_volumes[-1])
        reward += self.place_and_update_position(order)
        # strategic agent trades at some frquency, e.g. every 50 time steps
        if self.strategic_agent is not None:
            order_list = self.strategic_agent.generate_order(self.time, lob=self.lob)
            reward += self.place_and_update_position(order_list)
        self.time += 1
        assert self.execution_agent.volume >= 0
        return reward

    def reset(self, seed=None, options={}):
        super().reset(seed=seed)
        # reset strategic agent 
        if self.strategic_agent is not None:
            self.strategic_agent.reset()
        # reset execution agent, this will set the direction for the execution agent 
        self.execution_agent.reset()
        # time of the simulation 
        self.time = -50
        # initialize the order book and register agents 
        if self.strategic_agent is None:
            list_of_agents = [self.execution_agent.agent_id, self.noise_agent.agent_id]
        else:
            list_of_agents = [self.execution_agent.agent_id, self.noise_agent.agent_id, self.strategic_agent.agent_id]
        self.lob = LimitOrderBook(level=self.noise_agent.level, list_of_agents = list_of_agents)
        # fill the book with initial limit orders 
        orders = self.noise_agent.initialize()
        self.lob.process_order_list(orders)        
        # run 50 transitions 
        for _ in range(50):
            out = self.transition()
        # return the observation
        observation = self.execution_agent.get_observation(self.time, self.lob)
        reward = 0
        return observation, {}
    
    def place_and_update_position(self, order):
        if order is None:
            return 0
        if type(order) == list:
            msg_list = [self.lob.process_order(o) for o in order]
            reward = self.execution_agent.update_position_from_message_list(msg_list)
        else:
            msg = self.lob.process_order(order)
            reward = self.execution_agent.update_position(msg)
        return reward
    
    def step(self, action=None):
        reward = 0 
        # orders by execution agent
        if self.execution_agent.agent_id == 'rl_agent':
            order = self.execution_agent.generate_order(self.time, self.lob, action)
        else:
            order = self.execution_agent.generate_order(self.time, self.lob)        
        # process those orders: the function places the orders in the book and updates the agents position
        reward += self.place_and_update_position(order)
        # agent can reach terminal condition by sending market orders 
        if self.execution_agent.volume == 0:                
            observation = self.execution_agent.get_observation(self.time, self.lob)
            return observation, reward, True, False, self.final_info() 
        # go through 100 steps by the noise traders and strategic agent
        for _ in range(100):
            reward += self.transition()
            if self.execution_agent.volume == 0:
                observation = self.execution_agent.get_observation(self.time, self.lob)
                return observation, reward, True, False, self.final_info()
        # handle terminal conditions 
        if self.time == self.terminal_time:
            order_list = []
            for order_id in self.lob.order_map_by_agent[self.execution_agent.agent_id]:
                order_list.append(Cancellation(agent_id=self.execution_agent.agent_id, order_id=order_id))
            order_list.append(MarketOrder(agent_id=self.execution_agent.agent_id, side='bid', volume=self.execution_agent.volume))
            reward += self.place_and_update_position(order_list)
            observation = self.execution_agent.get_observation(self.time, self.lob)
            assert self.execution_agent.active_volume == 0
            assert self.execution_agent.volume == 0, 'agent should have zero volume at terminal time'
            return observation, reward, True, False, self.final_info()
        observation = self.execution_agent.get_observation(self.time, self.lob)
        return observation, reward, False, False, {}
    
    def final_info(self):
        return {'total_reward': self.execution_agent.cummulative_reward, 'time': self.time, 'volume': self.execution_agent.volume, 'initial_volume': self.execution_agent.initial_volume, 'initial_bid': self.execution_agent.reference_bid_price, 'limit_sell':self.execution_agent.limit_sells, 'market_sell': self.execution_agent.market_sells}


config = {'seed':0, 'type':'noise', 'execution_agent':'rl_agent', 'terminal_time':int(1e3), 'volume':40, 'level':30, 'damping_factor':0.5, 'market_volume':2, 'limit_volume':5, 'frequency':50}  


if __name__ == '__main__': 
    config = {'seed':0, 'type':'noise', 'execution_agent':'rl_agent', 'terminal_time':int(1e3), 'volume':40, 'level':30, 'damping_factor':0.5, 'market_volume':2, 'limit_volume':5, 'frequency':50}   
    M = Market(config)
    for n in range(2):
        r = 0 
        print(f'episode {n}')
        M.reset()
        terminated = False 
        while not terminated:
            action = M.action_space.sample()
            observation, reward, terminated, truncated, info = M.step(action)
            r += reward
            # print(f'observation: {observation}') 
        print(f'reward: {r}')
        # print(f"info: {info['total_reward']}")
        # print(f'termindated: {terminated}')
        print(f'info: {info}')
    data, orders = M.lob.log_to_df()
    heat_map(orders, data, max_level=5, max_volume=50, scale=500)
    plt.show()

