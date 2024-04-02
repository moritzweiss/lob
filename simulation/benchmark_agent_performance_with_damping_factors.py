import pandas as pd
import numpy as np
from multiprocessing import Pool
import itertools
import time
from test_vectorization import mp_rollout
from all_markets_simulation import config 
env_config = config.copy()

if __name__ == '__main__':

    max_steps = int(1e3)
    volumes = [10, 40]
    strategies = ['market_agent', 'sl_agent', 'linear_sl_agent']
    env_types = ['noise', 'flow', 'strategic']
    damping_factors = [1.0]
    n_samples = 1000
    n_workers = 50



    start = time.time()

    for e_t in env_types:
        data = {}
        print('-------------------')
        print(f'env_type: {e_t}')
        print('-------------------')
        for strategy in strategies: 
            # for damping_factor in damping_factors:
            print('---')                  
            print(f'strategy: {strategy}')
            print('---')
            # data[tag] = []
            # tag = f'{e_t}_{strategy}'
            data[f'{strategy}_mean'] = []
            data[f'{strategy}_std'] = []
            for volume in volumes:
                rewards = mp_rollout(n_samples, n_workers, strategy, e_t, 1.0, volume)     
                data[f'{strategy}_mean'].append(np.mean(rewards))       
                data[f'{strategy}_std'].append(np.std(rewards))       
        
        data = pd.DataFrame.from_dict(data)
        data.index = volumes
        data.index.name = 'lots'
        data = data.round(2)
        print(data)
        # print(data)
        # print(f'total time: {round(time.time()-start,1)} seconds')
        data.to_csv(f'results/performance_benchmarks_{e_t}_c_{1}_sqrt.csv', index=True, float_format="%.2f")


        
        






        
