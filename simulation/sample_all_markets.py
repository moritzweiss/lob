from os import environ
import warnings 
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
        filled = []
        for _ in range(self.n_samples):
            terminated = False
            M.reset()
            while not terminated:
                terminated, info = M.step()
                if terminated:
                    break
            if info['volume'] == 0:
                filled.append(1 if info['time'] < self.terminal_time else 0)
                rewards.append(info['total_reward'])    
            elif info['volume'] > 0:   
                print('volume not fullly executed')
            elif info['volume'] < 0:   
                raise ValueError('volume should not be negative')
        return rewards, filled
    
    def mp_sample(self, n_workers=1,seed=1):
        seeds = [seed + i for i in range(n_workers)]
        # p = Pool(n_workers)
        # p.close()
        with Pool(n_workers) as pool:
            results, filled = pool.map(self.sample, seeds)
        return results, filled 

if __name__ == '__main__':
    # SM = SampleMarket(terminal_time=int(1e3), n_samples=10, level=30, volume=40, imbalance_reaction=True, damping_factor=0.75, strategy='sl', strategic_investor=True, market_volume=1, limit_volume=5)
    # rewards = SM.sample(0)
    # print(np.mean(rewards))
    
    seed = 0 
    total_samples = int(1e3)
    # total_samples = int(1e1)
    n_workers  = 70
    n_samples = int(np.ceil(total_samples/n_workers))    
    damping_factors = [0, 0.1, 0.25, 0.5, 1.0]
    print(f'n_workers: {n_workers}')
    print(f'n_samples per worker: {n_samples}')
    print(f'total_samples: {n_workers*n_samples}')
    terminal_time = int(1e3)
    volumes = [1, 10, 20, 40, 80]
    # volumes = [1, 10]
    # volumes = [10]
    strategies = ['market', 'sl', 'linear_sl']
    # strategies = ['sl', 'linear_sl']
    # market_environments = ['noise', 'flow', 'strategic']
    market_environments = ['noise', 'flow']
    
    # sample noise environment 
    start = time.time()
    seeds = [seed + i for i in range(n_workers)]#
    with Pool(n_workers) as pool:
        data = {}        
        for s in strategies:     
            print(f'strategy: {s}')  
            data[f'{s}_m'] = []
            data[f'{s}_std'] = []
            for v in volumes:                        
                me = 'noise'
                imbalance_reaction = False
                strategic_investor = False
                if v >= 10:
                    SM = SampleMarket(terminal_time=terminal_time, n_samples=n_samples, level=30, volume=v, imbalance_reaction=imbalance_reaction, strategy=s, strategic_investor=strategic_investor, market_volume=1, limit_volume=5)
                    out  = pool.map(SM.sample, seeds)
                    rewards = [o[0] for o in out]
                    rewards = list(itertools.chain.from_iterable(rewards))
                    # filled = [o[1] for o in out]
                    # filled = list(itertools.chain.from_iterable(filled))
                    # print(f'market environment: {me}, volume: {v}, strategy: {s}, mean reward: {np.mean(rewards)}')
                    data[f'{s}_m'].append(np.mean(rewards))
                    data[f'{s}_std'].append(np.std(rewards))
                else:
                    if s == 'market' or s == 'sl':
                        SM = SampleMarket(terminal_time=terminal_time, n_samples=n_samples, level=30, volume=v, imbalance_reaction=imbalance_reaction, strategy=s, strategic_investor=strategic_investor, market_volume=1, limit_volume=5)
                        out  = pool.map(SM.sample, seeds)
                        rewards = [o[0] for o in out]
                        rewards = list(itertools.chain.from_iterable(rewards))
                        # filled = [o[1] for o in out]
                        # filled = list(itertools.chain.from_iterable(filled))
                        # print(f'market environment: {me}, volume: {v}, strategy: {s}, mean reward: {np.mean(rewards)}')
                        data[f'{s}_m'].append(np.mean(rewards))
                        data[f'{s}_std'].append(np.std(rewards))
                    elif s == 'linear_sl':
                        data[f'{s}_m'].append(np.nan)
                        data[f'{s}_std'].append(np.nan)
                    else:
                        raise ValueError('strategy not found')


        #########
        # print(data)
        df = pd.DataFrame.from_dict(data)    
        df.index = volumes
        df.index.name = 'lots'
        df = df.round(2)
        print(df)
        # df.to_csv(f'./results/performance_benchmarks_{me}_environment_damping_factor_{damping_factor}_sqrt_imb.csv')
        df.to_csv(f'./results/performance_benchmarks_{me}_environment.csv')
        print('time for the noise environemnt: ', time.time() - start)



    # sample flow environment
    market_environments = ['flow', 'strategic']
    # market_environments = ['flow']
    seeds = [seed + i for i in range(n_workers)]
    with Pool(n_workers) as pool:
        for damping_factor in damping_factors:
            print('#######################')
            print(f'damping factor: {damping_factor}')
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
                        if v >= 10:
                            SM = SampleMarket(damping_factor=damping_factor, terminal_time=terminal_time, n_samples=n_samples, level=30, volume=v, imbalance_reaction=imbalance_reaction, strategy=s, strategic_investor=strategic_investor, market_volume=1, limit_volume=5)
                            out  = pool.map(SM.sample, seeds)
                            rewards = [o[0] for o in out]
                            rewards = list(itertools.chain.from_iterable(rewards))
                            # filled = [o[1] for o in out]
                            # filled = list(itertools.chain.from_iterable(filled))
                            # print(f'market environment: {me}, volume: {v}, strategy: {s}, mean reward: {np.mean(rewards)}')
                            data[f'{s}_m'].append(np.mean(rewards))
                            data[f'{s}_std'].append(np.std(rewards))
                        else:
                            if s == 'market' or s == 'sl':
                                SM = SampleMarket(damping_factor=damping_factor, terminal_time=terminal_time, n_samples=n_samples, level=30, volume=v, imbalance_reaction=imbalance_reaction, strategy=s, strategic_investor=strategic_investor, market_volume=1, limit_volume=5)
                                out  = pool.map(SM.sample, seeds)
                                rewards = [o[0] for o in out]
                                rewards = list(itertools.chain.from_iterable(rewards))
                                # filled = [o[1] for o in out]
                                # filled = list(itertools.chain.from_iterable(filled))
                                # print(f'market environment: {me}, volume: {v}, strategy: {s}, mean reward: {np.mean(rewards)}')
                                data[f'{s}_m'].append(np.mean(rewards))
                                data[f'{s}_std'].append(np.std(rewards))
                            elif s == 'linear_sl':
                                data[f'{s}_m'].append(np.nan)
                                data[f'{s}_std'].append(np.nan)
                            else:
                                raise ValueError('strategy not found')
                #########
                # print(data)
                df = pd.DataFrame.from_dict(data)    
                df.index = volumes
                df.index.name = 'lots'
                df = df.round(2)
                print(df)

                # df.to_csv(f'./results/performance_benchmarks_{me}_environment_damping_factor_{damping_factor}_sqrt_imb.csv')
                df.to_csv(f'./results/performance_benchmarks_{me}_environment_damping_factor_{damping_factor}.csv')
                print('time for this environemnt: ', time.time() - start)