from market_gym import mp_rollout
import numpy as np
import time
import pandas as pd 


# parameters 
n_samples = 1000
n_cpus = 70 # on ada-23, 70 cores seems roughly optimal 
seed = 100
execution_agent = 'sl_agent'
market_type = 'noise'
results = {}
# envs = ['noise', 'flow']
envs = ['noise', 'flow']
agents = ['sl_agent', 'linear_sl_agent']
lots = [20, 40]
# agents = ['sl_agent']


# run benchmarks 
start_time = time.time()
for market_type in envs:    
    print(f"Running benchmarks for {market_type} market")
    results = {agent: [] for agent in agents}
    # print(results)
    for agent in agents: 
        # print(agent)
        for n_lots in lots:
            rewards, times, n_events = mp_rollout(n_samples=n_samples, n_cpus=n_cpus, execution_agent=agent, market_type=market_type, volume=n_lots, seed=seed)
            # print(np.mean(rewards))
            results[agent].append(np.mean(rewards))
    results = pd.DataFrame(results).round(2)
    results.index = lots
    results.index.name = 'lots'
    results.to_csv(f'tables/{market_type}_results.csv')
    print(results)

# print execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

