from all_markets_simulation import Market
from multiprocessing import Pool
import numpy as np
import itertools
import time 
import pandas as pd 
from limit_order_book.limit_order_book import LimitOrderBook


def rollout(seed, num_episodes, execution_agent, market_type, volume):
    config = {}
    config['seed'] = seed
    config['terminal_time'] = int(1e3)
    config['frequency'] = 100
    config['volume'] = volume
    config['market_type'] = market_type
    config['execution_agent'] = execution_agent
    env = Market(config)
    total_rewards = []
    execution_time = []
    for _ in range(num_episodes):
        env.reset()
        total_reward = 0
        terminated = False
        while not terminated:
            # action = env.action_space.sample()  
            observation, reward, terminated, truncated, info = env.step()
            total_reward += reward
        total_rewards.append(total_reward)
        execution_time.append(info['time']-1)
        # print(info['time'])
    return total_rewards, execution_time 


def mp_rollout(n_samples, n_cpus, execution_agent, market_type, volume):
    samples_per_env = int(n_samples/n_cpus) 
    # print(f'starting {n_envs} workers')
    # print(f'{samples_per_env} per worker')
    with Pool(n_cpus) as p:
        out = p.starmap(rollout, [(seed, samples_per_env, execution_agent, market_type, volume) for seed in range(n_cpus)])    
    all_rewards, execution_time = zip(*out)
    all_rewards = list(itertools.chain.from_iterable(all_rewards))
    execution_time = list(itertools.chain.from_iterable(execution_time))
    return all_rewards, execution_time 


if __name__ == '__main__':
    rollout(seed=0, num_episodes=5, execution_agent='market_agent', market_type='flow', volume=1)
    n_samples = 1000
    n_cpus = 60
    start = time.time()
    # all_rewards = rollout(1, 10, 'market_agent', 'strategic', 40) 
    # for lots in [10, 20, 40, 60]:
    for lots in [10, 20]:
        print(f'{lots} lots')
        results = {}
        execution_times = {}
        for agent in ['market_agent', 'sl_agent', 'linear_sl_agent']:
        # for agent in ['market_agent', 'sl_agent']:
            execution_times[f'{agent}_mean'] = []
            execution_times[f'{agent}_std'] = []
            results[f'{agent}_mean'] = []
            results[f'{agent}_std'] = []
            for env in ['noise', 'flow']:
                all_rewards, execution_time = mp_rollout(n_samples, n_cpus, agent, env, lots)
                # print(agent)
                # print(env)
                # print(f'length: {len(all_rewards)}')
                # print(f'mean reward: {np.mean(all_rewards)}')
                # print(f'std reward: {np.std(all_rewards)}')             
                results[f'{agent}_mean'].append(np.mean(all_rewards))
                results[f'{agent}_std'].append(np.std(all_rewards))
                execution_times[f'{agent}_mean'].append(np.mean(np.array(execution_time)))
                execution_times[f'{agent}_std'].append(np.std(np.array(execution_time)))
        results = pd.DataFrame.from_dict(results)
        results.index = ['noise', 'flow']
        results = results.round(2)        
        print(results)
        results.to_csv(f'results/performance_benchmarks_{lots}.csv', index=True, float_format="%.2f")
        # print(execution_times)
        execution_times = pd.DataFrame.from_dict(execution_times)
        execution_times.index = ['noise', 'flow']
        execution_times = execution_times.round(2)
        print(execution_times)
    print('time',   time.time() - start)

