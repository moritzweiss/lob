from advanced_multi_lot import Market
import matplotlib.pyplot as plt
import numpy as np

config = {'total_n_steps': int(1e3), 'log': True, 'seed':10, 'initial_level': 4, 'initial_volume': 250, 'imbalance_trader': True}

# heat map 

M = Market(config)
M.initialize_book()

observation, _ = M.reset()
terminated = truncated = False 
while not terminated and not truncated: 
    action = np.array([-10, 10, -10, -10], dtype=np.float32)
    observation, reward, terminated, truncated, info = M.step(action)

plt.figure()
M.heat_map()
plt.savefig('plots/heat_map.png')

# hist of lots trade per 1000 steps 
M = Market(config)
M.initialize_book()
N = int(1e5)
market_orders = []
for n in range(N):
    if n%10000 == 0:
        print(n)
    M.logging()
    out = M.generate_order()
    # print(out)
    M.log_trade(out)
    if out[0] == 'market':
        market_orders.append(out[1]['market_agent'][0]['volume'])
print(sum(market_orders)/N)