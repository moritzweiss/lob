from os import environ
N_THREADS = '1'
environ['OMP_NUM_THREADS'] = N_THREADS
environ['OPENBLAS_NUM_THREADS'] = N_THREADS
environ['MKL_NUM_THREADS'] = N_THREADS
environ['VECLIB_MAXIMUM_THREADS'] = N_THREADS
environ['NUMEXPR_NUM_THREADS'] = N_THREADS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from agents import NoiseAgent
import gymnasium as gym 
from limit_order_book.limit_order_book import LimitOrderBook, MarketOrder, LimitOrder
from multiprocessing import Pool
import itertools
import time


class Market(gym.Env):
    def __init__(self, seed, n_steps=int(1e3), level=30, volume=10, side='ask', imbalance_reaction=False, linear_sl=False, damping_factor=0.0) -> None:
        super().reset(seed=seed)
        self.imbalance_reaction = imbalance_reaction
        self.n_steps = n_steps
        self.level = level
        self.noise_agent = NoiseAgent(level=level, rng=self.np_random, imbalance_reaction=self.imbalance_reaction, initial_shape_file='data_small_queue.npz', config_n=1, damping_factor=damping_factor)
        self.start_volume = volume 
        self.volume = volume
        self.side = side
        self.linear_submit_and_leave = linear_sl
        self.posted_volume = 0
        return None 
    
    def reset(self):
        self.volume = self.start_volume
        self.filled_volume = 0
        self.reward = 0 
        self.lob = LimitOrderBook(level=self.level, list_of_agents = ['smart_agent', 'noise_agent']) 
        self.time = 0
        orders = self.noise_agent.initialize()
        [self.lob.process_order(order) for order in orders]
        # submission by the sl agent         
        if self.linear_submit_and_leave:
            volume = int(self.start_volume/10)
        else:
            volume = self.start_volume
        if self.side == 'ask':
            price = self.lob.get_best_price('ask')
            order = LimitOrder('smart_agent', 'ask', price, volume)
            self.posted_volume += volume
        else:
            price = self.lob.get_best_price('bid')
            order = LimitOrder('smart_agent', 'bid', price, volume)
        self.lob.process_order(order)
        self.time = 0         
        if self.linear_submit_and_leave:
            self.volume_slice = int(self.start_volume/10)
            assert self.volume_slice * 10 == self.start_volume
        return None
    
    def noise_transition(self):
        terminated = False
        order = self.noise_agent.sample_order(self.lob.data.best_bid_prices[-1], self.lob.data.best_ask_prices[-1], self.lob.data.bid_volumes[-1], self.lob.data.ask_volumes[-1])
        out = self.lob.process_order(order)
        if order.type == 'market':
            if out.filled_orders['smart_agent']:
                for orders in out.filled_orders['smart_agent']:
                    filled_volume = orders['filled_volume' ]
                    self.volume -= filled_volume
                    self.filled_volume += filled_volume
                    fill_price = orders['fill_price']
                    self.reward += filled_volume * fill_price
                    if not self.linear_submit_and_leave:
                        v = [self.lob.order_map[id].volume for id in self.lob.order_map_by_agent['smart_agent']]
                        assert sum(v) == self.volume
                    # []: False, not [] : True : list is empty  
                    # self.lob.process_order(out[0])
                    if self.volume == 0:
                        # print(f'time: {self.time}')
                        # print(f'remaining volume: {self.volume}')
                        assert not self.lob.order_map_by_agent['smart_agent']
                        assert self.filled_volume == self.start_volume
                        terminated = True                    
                        return terminated
        return terminated
    
    def step(self):
        # noise transitions 
        for _ in range(100):
            terminated = self.noise_transition()
            if terminated:
                return terminated
            self.time += 1        
        if self.time == self.n_steps:
            # print(f'time: {self.time}')
            # print(f'remaining volume: {self.volume}')
            # set not empty 
            if self.lob.order_map_by_agent['smart_agent']:
                assert self.volume == self.start_volume - self.filled_volume
                market_order = MarketOrder('smart_agent', 'bid', self.volume)
                out = self.lob.process_order(market_order)
                self.volume = 0
                self.filled_volume = self.start_volume - self.filled_volume
                self.volume = 0
                self.reward += out.execution_price 
                terminated = True                    
                return terminated
        if self.linear_submit_and_leave:
            volume = self.volume_slice
            price = self.lob.get_best_price('ask')
            order = LimitOrder('smart_agent', 'ask', price, volume)
            self.lob.process_order(order)
            self.posted_volume += volume
        return terminated
    
    def market_order_payoff(self): 
        order = MarketOrder('smart_agent', 'bid', self.volume)
        out = self.lob.process_order(order)
        return (out.execution_price - self.volume*1000)/self.volume


class SampleMarket:
    def __init__(self, n_steps=int(1e3), n_samples=10, level=30, volume=10, side='ask', imbalance_reaction=False, place_at=1, initial_shape_file='data_small_queue.npz', linear_sl=False, damping_factor=0.0) :
        self.damping_factor = damping_factor
        self.level = level
        self.n_steps = n_steps
        self.n_samples = n_samples
        self.volume = volume 
        self.imbalance_reaction = imbalance_reaction
        self.linear_sl = False
        self.side = 'ask'
        self.linear_sl = linear_sl
        return None
    def sample(self, seed):
        M = Market(seed=seed, imbalance_reaction=self.imbalance_reaction,  n_steps=self.n_steps, level=self.level, volume=self.volume, side=self.side, linear_sl=self.linear_sl, damping_factor=self.damping_factor)
        rewards = []        
        for _ in range(self.n_samples):
            terminated = False
            M.reset()
            while not terminated:
                terminated = M.step()
                if terminated:
                    break
            rewards.append((M.reward-1000*self.volume)/self.volume)
        return rewards
    
    def mp_sample(self, n_workers=1,seed=1):
        seeds = [seed + i for i in range(n_workers)]
        # p = Pool(n_workers)
        with Pool(n_workers) as pool:
            results = pool.map(self.sample, seeds)
        # p.close()
        return results


if __name__ == '__main__':

    # variables 
    # 1000
    total_samples = int(1e3)
    # total_samples = int(5e2)
    n_workers  = 70
    n_samples = int(np.ceil(total_samples/n_workers))    
    print(f'n_workers: {n_workers}')
    print(f'n_samples: {n_samples}')
    print(f'total_samples: {n_workers*n_samples}')
    max_steps = int(1e3)
    # volumes = [1]
    volumes = [10, 20, 30, 40]
    volumes = [10, 20, 30, 40]
    # volumes = [10, 40]
    # volumes = [20, 30]
    strategies = ['market', 'sl', 'linear_sl']
    # strategies = ['sl']
    reactions = [False, True]
    # volumes = [10]

    

    start = time.time()

    for damping_factor in [0.25, 0.5, 1.0]:
        data = {}
        for imbalance_reaction in reactions:
            print('-------------------')
            print(f'{"with_imbalance" if imbalance_reaction else "without_imbalance"}')      
            print('-------------------')
            for strategy in strategies: 
                print('---')                  
                print(f'strategy: {strategy}')
                tag = f'{"noise" if not imbalance_reaction else "flow"}_{strategy}'
                data[tag] = []
                # TODO: clean up market order payoff 
                for volume in volumes:
                    # SM.reset()                
                    if strategy == 'sl':
                        linear_sl = False
                    else:
                        linear_sl = True    
                    SM = SampleMarket(n_steps=max_steps, n_samples=n_samples, level=30, volume=volume, side='ask', imbalance_reaction=imbalance_reaction, linear_sl=linear_sl, damping_factor=damping_factor)
                    if strategy == 'market':
                        M = Market(seed=0, imbalance_reaction=imbalance_reaction, n_steps=max_steps, level=30, volume=volume, side='ask', linear_sl=linear_sl)
                        M.reset()
                        r = M.market_order_payoff()
                        data[tag].append(r)
                    else: 
                        # r = SM.sample(1)
                        # start = time.time()
                        r = SM.mp_sample(n_workers=n_workers, seed=0)
                        # print(f'time: {time.time()-start}')
                        # print(r[0])
                        # print(r[1])
                        # print(r[2])
                        r = list(itertools.chain.from_iterable(r)) 
                        # print(f'length of return list: {len(r)}')
                        # print(f'mean reward: {np.mean(r)}')
                        # print(f'std reward: {np.std(r)}')
                        data[tag].append(np.mean(r)) 
                        # print(r)
                    
                    # fig, ax = plt.subplots() 
                    # ax.hist(r, bins=np.arange(start=-5.5, stop=6.0, step=1.0), color='blue', align='mid') 
                    # if imbalance_reaction:
                    #     im = 'with'
                    # else:
                    #     im = 'without'
                    # ax.set_title(f'reward, {volume} lots, {im} imbalance, {strategy}')             
                    # ax.set_xlabel('reward') 
                    # ax.set_ylabel('frequency') 
                    # ax.set_xlim(0, max_steps) 
                    # ax.set_ylim(0, 400)
                    # ax.set_xlim(-5.5, 5.5)     
                    # plt.tight_layout()
                    # plt.savefig(f'plots/{tag}_{volume}_lots.pdf', dpi=400)
        
        data = pd.DataFrame.from_dict(data)
        data.index = volumes
        data.index.name = 'lots'
        data = data.round(2)
        print(data)
        print(f'total time: {round(time.time()-start,1)} seconds')
        data.to_csv(f'results/performance_benchmarks_c_{damping_factor}.csv', index=True, float_format="%.2f")


        
        






        
