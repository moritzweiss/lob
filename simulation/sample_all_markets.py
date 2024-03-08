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

from all_markets_simulation import Market

class SampleMarket:
    def __init__(self, terminal_time=int(1e3), n_samples=10, level=30, volume=10, imbalance_reaction=False, damping_factor=0.5, strategy='market', strategic_investor=False, market_volume=1, limit_volume=5) :
        self.damping_factor = damping_factor
        self.level = level
        self.terminal_time = terminal_time
        self.n_samples = n_samples
        self.volume = volume 
        self.imbalance_reaction = imbalance_reaction
        # 
        self.strategy = strategy
        self.strategic_investor = strategic_investor
        self.market_volume = market_volume
        self.limit_volume = limit_volume        
        return None
    def sample(self, seed):
        M = Market(seed=seed, strategic_investor=self.strategic_investor, imbalance_reaction=self.imbalance_reaction,  terminal_time=self.terminal_time, level=self.level, volume=self.volume, damping_factor=self.damping_factor, strategy=self.strategy, limit_volume=5, market_volume=1, frequency=50)
        rewards = []        
        for _ in range(self.n_samples):
            terminated = False
            M.reset()
            while not terminated:
                terminated, info = M.step()
                if terminated:
                    break
            if info['volume'] == 0:    
                rewards.append(info['total_reward'])
            elif info['volume'] > 0:
                pass
            else:
                raise ValueError('volume is negative')
        return rewards
    
    def mp_sample(self, n_workers=1,seed=1):
        seeds = [seed + i for i in range(n_workers)]
        # p = Pool(n_workers)
        with Pool(n_workers) as pool:
            results = pool.map(self.sample, seeds)
        # p.close()
        return results

if __name__ == '__main__':
    # SM = SampleMarket(terminal_time=int(1e3), n_samples=10, level=30, volume=40, imbalance_reaction=True, damping_factor=0.75, strategy='sl', strategic_investor=True, market_volume=1, limit_volume=5)
    # rewards = SM.sample(0)
    # print(np.mean(rewards))
    
    seed = 0 
    total_samples = int(1e3)
    n_workers  = 70
    n_samples = int(np.ceil(total_samples/n_workers))    
    damping_factor = 1.0
    print(f'n_workers: {n_workers}')
    print(f'n_samples per worker: {n_samples}')
    print(f'total_samples: {n_workers*n_samples}')
    terminal_time = int(1e3)
    volumes = [10, 20, 30, 40, 100]
    strategies = ['market', 'sl', 'linear_sl']
    market_environments = ['noise', 'flow', 'strategic']
    market_environments = ['noise', 'flow']

    # start a pool of workers 
    seeds = [seed + i for i in range(n_workers)]#
    with Pool(n_workers) as pool:
        for me in market_environments:
            start = time.time()
            print('#######################')
            print(f'market environment: {me}')
            data = {}
            ######
            for s in strategies:            
                print(f'strategy: {s}')  
                data[f'{s}_m'] = []
                data[f'{s}_std'] = []
                for v in volumes:            
                    if me == 'noise':
                        imbalance_reaction = False
                        strategic_investor = False
                    if me == 'flow':
                        imbalance_reaction = True
                        strategic_investor = False
                    if me == 'strategic':
                        imbalance_reaction = True
                        strategic_investor = True
                    SM = SampleMarket(terminal_time=terminal_time, n_samples=n_samples, level=30, volume=v, imbalance_reaction=imbalance_reaction, damping_factor=damping_factor, strategy=s, strategic_investor=strategic_investor, market_volume=1, limit_volume=5)
                    rewards = pool.map(SM.sample, seeds)
                    rewards = list(itertools.chain(*rewards))
                    # print(f'market environment: {me}, volume: {v}, strategy: {s}, mean reward: {np.mean(rewards)}')
                    data[f'{s}_m'].append(np.mean(rewards))
                    data[f'{s}_std'].append(np.std(rewards))
            #########
            # print(data)
            df = pd.DataFrame.from_dict(data)    
            df.index = volumes
            df.index.name = 'lots'
            df = df.round(2)
            print(df)
            df.to_csv(f'./results/performance_benchmarks_{me}_environment_damping_factor_{damping_factor}_sgn_imb.csv')
            print('time for this environemnt: ', time.time() - start)