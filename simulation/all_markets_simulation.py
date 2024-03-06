from typing import Any
from agents import NoiseAgent
import gymnasium as gym 
from limit_order_book.limit_order_book import LimitOrderBook, MarketOrder, LimitOrder
from limit_order_book.plotting import heat_map
import matplotlib.pyplot as plt
import numpy as np


class StrategicAgent():
    def __init__(self, frequency, market_volume, limit_volume) -> None:
        self.frequency = frequency
        self.market_order_volume = market_volume
        self.limit_order_volume = limit_volume
        self.agent_id = 'strategic_agent'
        return None 
    
    def generate_order(self, time, best_bid, best_ask):
        if time % self.frequency == 0: 
            order_list = []
            order_list.append(MarketOrder(self.agent_id, 'bid', self.market_order_volume))
            order_list.append(LimitOrder(self.agent_id, 'ask', best_ask, self.limit_order_volume))                        
            return order_list
        else: 
            return None 

class BenchmarkAgent():
    def __init__(self, volume, terminal_time, frequency=100, strategy='market') -> None:
        assert strategy in ['market', 'linear_sl', 'sl']
        self.when_to_place = 0 
        self.initial_volume = volume
        self.volume = volume
        self.active_volume = 0
        self.agent_id = 'sl_agent'
        self.r = 0
        self.terminal_time = terminal_time
        self.passive_fills = 0
        self.market_fills = 0
        self.strategy = strategy
        self.frequency = frequency
        assert volume % 10 == 0 
        assert self.when_to_place < self.terminal_time
        self.volume_slice = int(self.initial_volume/10)
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
        self.r += (cash - volume * self.initial_bid) / self.initial_volume
        return None

    def update_position(self, fill_message):
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
            if fill_message.agent_id == self.agent_id:
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
                    cash += m.filled_volume * m.fill_price
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
    def __init__(self, seed, terminal_time=int(1e3), volume=20, level=30, imbalance_reaction=False, damping_factor=0.5, strategic_investor=False, market_volume=2, limit_volume=5, frequency=50, strategy='market') -> None:
        # total time steps to simulate
        self.terminal_time = terminal_time
        # initialize benchamrk agent here 
        self.execution_agent = BenchmarkAgent(volume=volume, terminal_time=terminal_time, strategy=strategy)
        # maybe levels into the config file always = 30 anwyays?
        self.noise_agent = NoiseAgent(level=level, rng=np.random.default_rng(seed) , imbalance_reaction=imbalance_reaction, initial_shape_file='data_small_queue.npz', config_n=1, damping_factor=damping_factor)
        if strategic_investor:
            self.strategic_agent = StrategicAgent(frequency=frequency, market_volume=market_volume, limit_volume=limit_volume)
        else:
            self.strategic_agent = None 
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
                # print(f'stratefic agent sends order at time {self.time}')
                # print(order_list)
                messages = [self.lob.process_order(order) for order in order_list]
                for m in messages:
                    if self.execution_agent.update_position(m):
                        return True
        self.time += 1
        return False
    
    def reset(self):
        # reset agent
        self.execution_agent.r = 0
        self.execution_agent.passive_fills = 0
        self.execution_agent.market_fills = 0
        self.execution_agent.volume = self.execution_agent.initial_volume
        self.execution_agent.active_volume = 0
        # 
        self.time = -50
        # initialize the order book, alternatively we could also implement a reset function for the order book 
        if self.strategic_agent is not None:
            self.lob = LimitOrderBook(level=self.noise_agent.level, list_of_agents = [self.execution_agent.agent_id, self.noise_agent.agent_id, self.strategic_agent.agent_id]) 
        else:
            self.lob = LimitOrderBook(level=self.noise_agent.level, list_of_agents = [self.execution_agent.agent_id, self.noise_agent.agent_id])
        # initialize book with noise agent 
        orders = self.noise_agent.initialize()
        [self.lob.process_order(order) for order in orders]
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
        # transition to next state 
        for _ in range(100):
            self.transition()
            # check if agent has zero volume
            if self.execution_agent.volume == 0:
                return True, self.final_info()
        return False, {}
    
    def final_info(self):
        return {'total_reward': self.execution_agent.r, 'time': self.time, 'volume': self.execution_agent.volume, 'initial_volume': self.execution_agent.initial_volume, 'passive':self.execution_agent.passive_fills, 'market': self.execution_agent.market_fills}


if __name__ == '__main__':
    # ToDO: implement benchmarks market, linear submit and leave 
    M = Market(seed=2, imbalance_reaction=True,  terminal_time=1000, volume=40, level=30, damping_factor=0.5, market_volume=1, limit_volume=5, frequency=50, strategic_investor=False, strategy='linear_sl')
    M.reset()
    # print(M.time)
    terminated = False 
    while not terminated:
        terminated, info = M.step()
    print(info)
    data, orders = M.lob.log_to_df()
    # heat_map(trades=orders, level2=data, max_level=5, max_volume=30, scale=500)
    # plt.show()

