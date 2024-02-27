from average_shape_vectorized import NoiseAgent, Market
from limit_order_book.plotting import heat_map, plot_average_book_shape
import time 
import matplotlib.pyplot as plt 
import pandas as pd 
pd.set_option('display.max_rows', 100)

level = 30
T = 1e3

start = time.time()

env = Market(level=level, seed=0, terminal_time=int(T))


for _ in range(5):
    env.reset()

    terminated = False 
    t = 1 
    while not terminated:
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if t%int(1e4) == 0:
            print(t)
        t += 1 

    print(f'time elapsed in minutes: {(time.time()-start)/60}')

    data, orders = env.lob.log_to_df()

    total = orders[orders.type == 'M']['size'].sum()
    print(f'total markert orders {total}')
    m = orders[orders.type == 'M']['size'].mean()
    print(f'mean markert orders {m}')

    heat_map(trades=orders, level2=data, max_level=5, scale=300, max_volume=50)

    # plot_average_book_shape(bid_volumes=env.lob.data.bid_volumes, ask_volumes=env.lob.data.ask_volumes, level=3, symetric=True)



plt.show()

print(1)


