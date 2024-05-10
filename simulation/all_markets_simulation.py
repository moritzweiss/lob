from os import environ
import os 
import sys 
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)
N_THREADS = '1'
environ['OMP_NUM_THREADS'] = N_THREADS
environ['OPENBLAS_NUM_THREADS'] = N_THREADS
environ['MKL_NUM_THREADS'] = N_THREADS
environ['VECLIB_MAXIMUM_THREADS'] = N_THREADS
environ['NUMEXPR_NUM_THREADS'] = N_THREADS
from agents import NoiseAgent, MarketAgent, SubmitAndLeaveAgent, LinearSubmitLeaveAgent, RLAgent, StrategicAgent
import gymnasium as gym 
from limit_order_book.plotting import heat_map, plot_prices
import matplotlib.pyplot as plt
from limit_order_book.limit_order_book import LimitOrderBook
import numpy as np
from gymnasium.spaces import Box
import os
from config.config import noise_agent_config, sl_agent_config, market_agent_config, rl_agent_config, l_sl_agent_config, strategic_agent_config

# TODO: test critical methods reward calculation and so on 
# TODO: redo testing of limit order book 
# Question: how does seeding work in rllib with multiple workers? MAKE SURE noise agent has its own specific seed 

class Market(gym.Env):
    """ 
    Attributes:
        - market_type: noise, flow, strategic, controls the type of environment
        - execution_agent: market_agent, sl_agent, linear_sl_agent, rl_agent, controls the type of execution agent
        - terminal_time: when to stop the environment, typically = 1000
        - frequency: frequency at which to act, typically = 100
        - noise_step: counts the steps of the noise agent (we use this as event time) 
        - physical_time: counts the physical time based on exponential distribution 
        - lob: limit order book instance 
        - action space
        - observation space 
        - drift: drift of the strategic agent. we add this manually to make training easier. this should be raplced with a proxy        
    """ 

    def __init__(self, config):                
        """
        Arguments:
            - mote: a config is used because this is preferred by rllib 
            - config: dictionary with the following keys
                - seed: seed for the environment 
                - market_type: noise, flow, strategic
                - execution_agent: market_agent, sl_agent, linear_sl_agent, rl_agent
                - terminal_time: when to stop the environment
                - volume: volume of the agent 
        Remark: could seperate into two agents. one market and one limit agent. 
        """
        # execution agent stuff: 
        if config['execution_agent'] == 'market_agent':
            # use market agent config here 
            market_agent_config['volume'] = config['volume']
            market_agent_config['terminal_time'] = config['terminal_time']
            self.execution_agent = MarketAgent(**market_agent_config)
        elif config['execution_agent'] == 'sl_agent':
            # use sl agent config here
            # update volume and terminal time 
            sl_agent_config['volume'] = config['volume']
            sl_agent_config['terminal_time'] = config['terminal_time']
            self.execution_agent = SubmitAndLeaveAgent(**sl_agent_config)
        elif config['execution_agent'] == 'linear_sl_agent':
            # use linear sl agent config here
            l_sl_agent_config['volume'] = config['volume']
            l_sl_agent_config['terminal_time'] = config['terminal_time']
            l_sl_agent_config['frequency'] = config['frequency']
            self.execution_agent = LinearSubmitLeaveAgent(**l_sl_agent_config)
        elif config['execution_agent'] == 'rl_agent':
            # use rl config here 
            rl_agent_config['volume'] = config['volume']
            rl_agent_config['terminal_time'] = config['terminal_time']
            self.execution_agent = RLAgent(**rl_agent_config)
        elif config['execution_agent'] == None:
            self.execution_agent = None   
        else:
            raise ValueError(f"Unknown value for execution agent {config['execution_agent']}")
        
        # setting market environment 
        assert config['market_type'] in ['noise', 'flow', 'strategic'], f'Unknown type {config["market_type"]}'
        self.market_type = config['market_type']

        # setting noise agent
        noise_agent_config['rng'] = np.random.default_rng(config['seed'])
        if config['market_type'] == 'noise':
            noise_agent_config['imbalance_reaction'] = False
            noise_agent_config['initial_shape_file'] = 'initial_shape/noise.npz'
        else:
            noise_agent_config['imbalance_reaction'] = True
            noise_agent_config['initial_shape_file'] = 'initial_shape/noise_flow_75.npz'
        # TODO: review whats happening here !! 
        # worker_index = getattr(config, 'worker_index', None)
        # if worker_index is not None:
        #     super().reset(seed=config['seed']+worker_index)
        # else:
        #     super().reset(seed=config['seed'])
        self.noise_agent = NoiseAgent(**noise_agent_config)
        
        # setting strategic agent
        if config['market_type'] == 'strategic':
            strategic_agent_config['rng'] = np.random.default_rng(config['seed'])
            self.strategic_agent = StrategicAgent(**strategic_agent_config)
        else:
            self.strategic_agent = None

        # terminal time
        self.terminal_time = config['terminal_time']

        # observation space is time and inventory 
        # 1: time, 2: inventory, 3:active volume, 4:inactive volume, 5: best_bid_drift, 6: spread, 7:shape of the book:, 8: own order distribution [l1, l2, l3, >l4], 9: inactive volume, imbalance 
        # 6 + 6 + 4 + 1 = 16
        if config['market_type'] == 'strategic':
            # add one observation for drift 
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(17+1,), seed=config['seed'], dtype=np.float32) 
        else:
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(17,), seed=config['seed'], dtype=np.float32) 

        # action space [m, l1, l2, l3, inactive]
        self.action_space = Box(low=-10, high=10, shape=(5,), seed=config['seed'], dtype=np.float32)

        # reset the environment 

        ## 
        # print(f'worker index is: {worker_index}')
        # not sure what this is doing. but whatever. lets keep this here for now. 
                
        self.physical_time = None  
        self.noise_step = None 
        self.lob = None 
        self.drift = None

        # whether to use physical time or event time to update the order book 
        # we can also use this to turn of eponential distribution sampling (maybe this will make the environent faster)
        self.use_physical_time = True 


    def reset(self, seed=None, options={}):
        """
        - reset physical time and noise step 
        - reset agents
        - initialize the order book, then 
        - run 50 transitions

        Returns:
            - observation (observation depends on the execution agent, sl agent just returns None, rl agent returns full observation)
            - {}: empty dictionary
        """
        super().reset(seed=seed)
        self.event_times = [0]
        self.physical_time = 0  
        self.noise_step = -50 
        # reset agents, register, initialize 
        if self.strategic_agent is not None:
            self.strategic_agent.reset()
            if self.strategic_agent.direction == 'sell':
                self.drift = -1
            elif self.strategic_agent.direction == 'buy':
                self.drift = 1
            else:
                raise ValueError(f'Unknown direction {self.strategic_agent.direction}')
        self.execution_agent.reset()
        if self.strategic_agent is None:
            list_of_agents = [self.execution_agent.agent_id, self.noise_agent.agent_id]
        else:
            list_of_agents = [self.execution_agent.agent_id, self.noise_agent.agent_id, self.strategic_agent.agent_id]
        self.lob = LimitOrderBook(level=self.noise_agent.level, list_of_agents = list_of_agents)
        orders = self.noise_agent.initialize(time=self.noise_step)
        reward, terminated = self.place_and_update(orders)
        assert terminated == False
        assert reward == 0
        # run transitions
        for _ in range(50):
            self.single_transition()
        observation = self._get_observation()
        return observation, {}
    

    def step(self, action=None):
        """
        run the environemnt until the next observation happens 

        Arguments:
            - action: if rl agent exists, action is a numpy array, otherwise it is None
        Returns:
            - observation: observation depends on the execution agent, sl agent just returns None, rl agent returns full observation
            - total_reward: total reward from 0 to 100 
            - terminated: True if the environment is terminated
            - truncated: True if the environment is truncated
            - info dict: output depends on whether the state is terminal or not 
        """
        assert self.noise_step >= 0
        total_reward = 0
        for _ in range(100):
            reward, terminated = self.single_transition(action)
            total_reward += reward
        if terminated: 
            assert self.execution_agent.volume == 0, 'agent should have zero volume'
            observation = self._get_observation()
            return observation, total_reward, True, False, self.final_info()
        elif self.noise_step < self.terminal_time:
            observation = self._get_observation()
            return observation, total_reward, False, False, {}
        else:
            raise ValueError(f'noise step is {self.noise_step} and terminal time is {self.terminal_time}')
    

    def single_transition(self, action=None):
        total_reward = 0
        # orders by execution agent 
        if self.execution_agent is not None:
            if self.execution_agent.agent_id == 'rl_agent':
                order = self.execution_agent.generate_order(self.noise_step, self.lob, action)
            else:
                order = self.execution_agent.generate_order(self.noise_step, self.lob)        
            reward, terminated = self.place_and_update(order)
            total_reward += reward
        # noise agent 
        order, waiting_time = self.noise_agent.sample_order(self.lob, self.noise_step)
        self.physical_time += waiting_time
        self.event_times.append(self.physical_time)
        reward, terminated = self.place_and_update([order])
        total_reward += reward
        # strategic agent
        if self.strategic_agent is not None:
            order_list = self.strategic_agent.generate_order(self.lob, self.noise_step)
            reward, terminated = self.place_and_update(order_list)
            total_reward += reward
        self.noise_step += 1 
        assert self.noise_step <= self.terminal_time, 'noise step should not exceed terminal time'
        # handle terminal condition 
        if self.noise_step == self.terminal_time and not terminated:
            order_list = self.execution_agent.sell_remaining_position(self.lob, self.noise_step)
            reward, terminated = self.place_and_update(order_list)
            assert terminated
            assert self.execution_agent.active_volume == 0
            assert self.execution_agent.volume == 0, 'agent should have zero volume at terminal time'
            total_reward += reward
        # log orders here         
        return total_reward, terminated


    def place_and_update(self, order):
        if order is None:
            return 0, False
        else:
            msg_list = [self.lob.process_order(o) for o in order]
            return self.execution_agent.update_position_from_message_list(msg_list)


    def _get_observation(self):
        if self.execution_agent.agent_id == 'rl_agent':
            if self.market_type == 'strategic':
                assert self.drift is not None
                observation = np.append(self.execution_agent.get_observation(self.noise_step, self.lob), self.drift).astype(np.float32)
            else:
                observation = self.execution_agent.get_observation(self.noise_step, self.lob)       
        else:
            observation = self.execution_agent.get_observation(self.noise_step, self.lob)
        return observation
    
    def final_info(self):
        bid = self.lob.get_best_price('bid')
        ask = self.lob.get_best_price('ask')
        return {'total_reward': self.execution_agent.cummulative_reward, 'time': self.noise_step, 'volume': self.execution_agent.volume, 'initial_volume': self.execution_agent.initial_volume, 
                'initial_bid': self.execution_agent.reference_bid_price, 'limit_sell':self.execution_agent.limit_sells, 'market_sell': self.execution_agent.market_sells, 
                'best_bid': bid, 'best_ask': ask}



if __name__ == '__main__': 
    config = {'seed':2, 'market_type':'noise', 'execution_agent':'sl_agent', 'terminal_time':int(3e2), 'volume':40}  
    print(config)
    M = Market(config)
    print('init is done')
    price_moves = []
    for n in range(1):
        r = 0 
        print(f'episode {n}')        
        observation, _ = M.reset()
        # assert observation in M.observation_space
        terminated = False 
        while not terminated:
            action = M.action_space.sample()
            action = np.array([0, 0, 0, 10, 0], dtype=np.float32)
            observation, reward, terminated, truncated, info = M.step(action)
            # print(M.noise_step)
            # print(M.physical_time)
            # print(observation[-2])
            # assert observation in M.observation_space
            # print(f'observation: {observation}')
            # print(f'order distribution: {M.execution_agent.volume_per_level}')
            price_moves.append((M.lob.get_best_price('ask')-1000)/10) 
            r += reward
            # print((M.lob.get_best_price('ask')-1000)/10)
            # print(f'observation: {observation}') 
            # print(f'reward: {r}')
            # print(f"info: {info['total_reward']}")
            # print(f'termindated: {terminated}')
        # print(f'info: {info}')
        if info["total_reward"] > 1:
            print(f'total reward: {info["total_reward"]}')
            print(info)

    print(info)            
    data, orders, market_orders = M.lob.log_to_df()
    # use physical time 
    # heat_map(market_orders, data, event_times=M.event_times, max_level=5, max_volume=40, scale=200)
    # use tick time 
    heat_map(market_orders, data, event_times=market_orders.time, max_level=5, max_volume=40, scale=1000)
    # plot_prices(level2=data, trades=orders, marker_size=200)
    plt.tight_layout()
    plt.savefig('heat.pdf')
    # plt.show()
    # print(max(price_moves))
    # print(min(price_moves)) 

