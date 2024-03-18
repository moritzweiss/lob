from typing import Any
from agents import NoiseAgent
import gymnasium as gym 
from limit_order_book.limit_order_book import LimitOrderBook, MarketOrder, LimitOrder
from limit_order_book.plotting import heat_map
import matplotlib.pyplot as plt
import numpy as np


class StrategicAgent():
    def __init__(self, frequency, market_volume, limit_volume, rng) -> None:
        self.frequency = frequency
        self.market_order_volume = market_volume
        self.limit_order_volume = limit_volume
        self.agent_id = 'strategic_agent'
        self.rng = rng
        self.direction = None 
        return None 
    
    def generate_order(self, time, best_bid, best_ask):
        if time % self.frequency == 0: 
            if self.direction == 'sell':
                order_list = []
                order_list.append(MarketOrder(self.agent_id, 'bid', self.market_order_volume))
                order_list.append(LimitOrder(self.agent_id, 'ask', best_ask, self.limit_order_volume))                        
                return order_list
            elif self.direction == 'buy':
                order_list = []
                order_list.append(MarketOrder(self.agent_id, 'ask', self.market_order_volume))
                order_list.append(LimitOrder(self.agent_id, 'bid', best_bid, self.limit_order_volume))                        
                return order_list
            else:
                raise ValueError(f'direction must be either buy or sell, got {self.direction}')
        else: 
            return None 
        
    def reset(self):
        self.direction = self.rng.choice(['buy', 'sell'])
        # self.direction = 'buy'
        # print(self.direction)
        return None

class BenchmarkAgent():
    def __init__(self, volume, terminal_time, frequency=100, strategy='market') -> None:
        assert strategy in ['market', 'linear_sl', 'sl']
        self.when_to_place = 0 
        self.frequency = frequency
        self.initial_volume = volume
        self.agent_id = 'sl_agent'
        self.terminal_time = terminal_time
        self.strategy = strategy
        assert volume % 10 == 0 or volume < 10
        assert self.when_to_place < self.terminal_time
        if volume >= 10:
            self.volume_slice = int(self.initial_volume/10)
            assert self.volume_slice * 10 == volume
        # self.no_action = no_action
        # self.reset()
        return None 
    
    def reset(self):
        self.volume = self.initial_volume
        self.active_volume = 0
        self.cummulative_reward = 0
        self.passive_fills = 0
        self.market_fills = 0
        return None
    
    def generate_order(self, time, best_bid, best_ask):
        if time == self.when_to_place:
            self.initial_bid = best_bid
        if self.strategy == 'sl':
            if time == self.when_to_place:
                return LimitOrder(self.agent_id, side='ask', price=best_ask, volume=self.volume)
            elif time == self.terminal_time: 
                return MarketOrder(self.agent_id, side='bid', volume=self.volume)
            else:
                return None
        elif self.strategy == 'linear_sl':
            if time % self.frequency == 0 and time < self.terminal_time:
                return LimitOrder(self.agent_id, side='ask', price=best_ask, volume=self.volume_slice)
            elif time == self.terminal_time:
                return MarketOrder(self.agent_id, side='bid', volume=self.volume)
            else:
                return None
        elif self.strategy == 'market':
            if time == self.when_to_place:
                return MarketOrder(self.agent_id, side='bid', volume=self.volume)
            else:
                return None

    
    def reward(self, cash, volume):        
        self.cummulative_reward += (cash - volume * self.initial_bid) / self.initial_volume
        return None
    
    def update_position_from_message_list(self, message_list):
        for m in message_list:
            if self.update_position(m): 
                return True
        return False


    def update_position(self, fill_message):
        # return True if agent has zero volume
        if fill_message.type == 'modification':
            # this agent doesnt modify orders
            pass
        elif fill_message.type == 'cancellation_by_price_volume':
            # this agent doesn cancel orders by price and volume
            pass 
        elif fill_message.type == 'limit':
            if fill_message.agent_id == self.agent_id:
                self.active_volume += fill_message.volume
        elif fill_message.type == 'cancellation':
            if fill_message.agent_id == self.agent_id:
                self.active_volume -= fill_message.volume
        elif fill_message.type == 'market':
            # check for potential fills 
            if fill_message.order.agent_id == self.agent_id:
                self.active_volume -= fill_message.filled_volume
                self.volume -= fill_message.filled_volume
                self.market_fills += fill_message.filled_volume
                self.reward(fill_message.execution_price, fill_message.filled_volume)
                if self.volume == 0:
                    return True                
            if self.agent_id in fill_message.passive_fills:
                cash = 0
                volume = 0 
                for m in fill_message.passive_fills[self.agent_id]:
                    volume += m.filled_volume
                    cash += m.filled_volume * m.order.price 
                self.active_volume -= volume 
                self.volume -= volume 
                self.passive_fills += volume
                self.reward(cash, volume)
                if self.volume == 0:
                    return True
            assert self.active_volume >= 0
            assert self.volume >= 0
            assert self.volume >= 0
        else: 
            raise ValueError(f'Unknown message type {fill_message.type}')
        return False 



class Market(gym.Env):
    def __init__(self, seed, type='noise', benchmark_agent='market', terminal_time=int(1e3), volume=20, level=30, damping_factor=0.5, market_volume=2, limit_volume=5, frequency=50) -> None:
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
        self.terminal_time = terminal_time
        # initialize benchmark agent here 
        if benchmark_agent is not None:
            self.execution_agent = BenchmarkAgent(volume=volume, terminal_time=terminal_time, strategy=benchmark_agent)
        else:
            self.execution_agent = None
        # types of market environments 
        if type == 'noise':
            self.noise_agent = NoiseAgent(level=level, rng=np.random.default_rng(seed) , imbalance_reaction=False, initial_shape_file='data_small_queue.npz', config_n=1, damping_factor=damping_factor)
            self.strategic_agent = None
        elif type == 'flow':
            self.noise_agent = NoiseAgent(level=level, rng=np.random.default_rng(seed) , imbalance_reaction=True, initial_shape_file='data_small_queue.npz', config_n=1, damping_factor=damping_factor)
            self.strategic_agent = None
        elif type == 'strategic':
            self.noise_agent = NoiseAgent(level=level, rng=np.random.default_rng(seed) , imbalance_reaction=True, initial_shape_file='data_small_queue.npz', config_n=1, damping_factor=damping_factor)
            self.strategic_agent = StrategicAgent(frequency=frequency, market_volume=market_volume, limit_volume=limit_volume, rng=np.random.default_rng(seed))
        return None 
    
    def transition(self):
        # noise trader trades at every time step        
        order = self.noise_agent.sample_order(self.lob.data.best_bid_prices[-1], self.lob.data.best_ask_prices[-1], self.lob.data.bid_volumes[-1], self.lob.data.ask_volumes[-1])
        msg = self.lob.process_order(order)
        if self.execution_agent.update_position(msg):
            return True
        # strategic agent trades at some frquency, e.g. every 50 time steps, can be turned off as well 
        if self.strategic_agent is not None:
            order_list = self.strategic_agent.generate_order(self.time, best_bid=self.lob.get_best_price('bid'), best_ask=self.lob.get_best_price('ask'))
            if order_list is not None:
                msgs = [self.lob.process_order(order) for order in order_list]
                if self.execution_agent.update_position_from_message_list(msgs):
                    return True
        self.time += 1
        return False
    
    def reset(self):
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
        return None 
    
    def step(self):
        # execution agent 
        order = self.execution_agent.generate_order(self.time, best_bid=self.lob.get_best_price('bid'), best_ask=self.lob.get_best_price('ask'))        
        if order is not None:
            msg = self.lob.process_order(order)        
            self.execution_agent.update_position(msg)
            # with a market order the agent could end up with zero volume 
            if self.execution_agent.volume == 0:
                return True, self.final_info() 
        if self.time == self.terminal_time:
            if self.execution_agent.volume > 0:
                print('agents position could not be fully executed')            
            return True, self.final_info()
        # transition to next state 
        for _ in range(100):
            self.transition()
            # check if agent has zero volume
            if self.execution_agent.volume == 0:
                return True, self.final_info()
        return False, {}
    
    def final_info(self):
        return {'total_reward': self.execution_agent.cummulative_reward, 'time': self.time, 'volume': self.execution_agent.volume, 'initial_volume': self.execution_agent.initial_volume, 'passive':self.execution_agent.passive_fills, 'market': self.execution_agent.market_fills}


if __name__ == '__main__':
    # ToDO: implement benchmarks market, linear submit and leave 
    M = Market(seed=2, type='flow', benchmark_agent='sl', volume=10)
    for _ in range(10):
        M.reset()
        terminated = False 
        while not terminated:
            terminated, info = M.step()
        print(info)
        data, orders = M.lob.log_to_df()

