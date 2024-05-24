import sys
import os
# import mat
# Add parent directory to python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import time 
from agents import NoiseAgent
import gymnasium as gym 
from limit_order_book.limit_order_book import LimitOrderBook
from limit_order_book.limit_order_book import LimitOrder
from limit_order_book.plotting import heat_map
from multiprocessing import Pool
from numpy.random import default_rng
import numpy as np


# ToDo: write a test for limit order fills by agent: place order somewhere in the book and check if it is filled
# make fill messages consistent: the same for every order type ! 
# ToDo: write a test for fill time of limit orders in a deterministic setting 


class Market(gym.Env):
    def __init__(self, rng, n_steps=int(1e3), config_n=1, n_samples=10, level=30, volume=10, side='ask', imbalance_reaction=False, place_at=1, initial_shape_file='data_small_queue.npz', damping_factor=0.5) -> None:
        self.noise_agent = NoiseAgent(level=level, rng=rng, config_n=config_n, imbalance_reaction=imbalance_reaction, initial_shape_file=initial_shape_file, damping_factor=damping_factor)
        self.level = level
        self.n_steps = n_steps
        self.volume = volume 
        self.side = side
        self.place_at = place_at
        # self.damping_factor = damping_factor
        # self.n_samples = n_samples
        return None 
    
    def reset(self):
        self.lob = LimitOrderBook(level=self.level, list_of_agents=['noise_agent', 'smart_agent'])
        self.time = 0
        orders = self.noise_agent.initialize()
        [self.lob.process_order(order) for order in orders]
        if self.side == 'ask':
            price = self.lob.get_best_price('ask')
            order = LimitOrder('smart_agent', 'ask', price+self.place_at, self.volume)
        else:
            price = self.lob.get_best_price('bid')
            order = LimitOrder('smart_agent', 'bid', price, self.volume)
        self.lob.process_order(order) 
        self.time = 0 
        return None
    
    def step(self):
        terminated = False
        order = self.noise_agent.generate_order(self.lob.data.best_bid_prices[-1], self.lob.data.best_ask_prices[-1], self.lob.data.bid_volumes[-1], self.lob.data.ask_volumes[-1])
        # print(order.type)
        out = self.lob.process_order(order)
        if out.type == 'market':
            if 'smart_agent' in out.passive_fills:
                if not self.lob.order_map_by_agent['smart_agent']:
                    terminated = True
                    return terminated
                    # self.lob.process_order(out[0])
        self.time += 1
        return terminated
    


class SampleMarket:
    def __init__(self, n_steps=int(1e3), config_n=1, n_samples=10, level=30, volume=10, side='ask', imbalance_reaction=False, place_at=1, initial_shape_file='data_small_queue.npz', damping_factor=0.5) :
        # self.noise_agent = NoiseAgent(level=level, rng=rng, config_n=config_n, imbalance_reaction=imbalance_reaction, initial_shape_file=initial_shape_file)
        self.level = level
        self.n_steps = n_steps
        self.volume = volume 
        self.side = side
        self.place_at = place_at
        self.n_samples = n_samples
        self.imbalance_reaction = imbalance_reaction
        self.damping_factor = damping_factor
        return None
    def sample(self, seed):
        M = Market(rng=default_rng(seed), imbalance_reaction=self.imbalance_reaction,  n_steps=self.n_steps, n_samples=self.n_samples, level=self.level, volume=self.volume, side=self.side, place_at=self.place_at, damping_factor=self.damping_factor)
        fill_times = []        
        for _ in range(self.n_samples):
            order_filled = False
            M.reset()
            for t in range(M.n_steps):
                order_filled = M.step()
                if order_filled:
                    break 
            fill_times.append(t)
        return fill_times
    def mp_sample(self, n_workers, seed=0):
        seeds = [seed+i for i in range(n_workers)]
        p = Pool(n_workers)
        fill_times = p.map(self.sample, seeds)
        return fill_times
    

if __name__ == '__main__':

    n_workers = 50
    n_samples = int(20)
    max_steps = int(1e3)
    # placed_at = [0, 1, 2, 3]
    placed_at = [0, 1, 2]
    # damping_factors = [0.0, 0.5, 1.0, 2.0]
    # damping_factors = ['no_imbalance', 0.0, 0.1, 0.25, 0.5, 1.0]
    damping_factors = ['no_imbalance', 0.1, 0.25, 0.5, 1.0]
    volumes = [1, 10, 20, 40, 80]
    
    for volume in volumes:
        print('#######')
        print(f'volume {volume}')
        data = {}   
        for d in damping_factors: 
            print(f'damping factor {d}')
            column_name = f'{d}'
            data[column_name] = []
            for place in placed_at: 
                print(f'placed at {place}')
                # loop over imbalance reaction and volume 
                if d == 'no_imbalance':
                    SM = SampleMarket(n_steps=max_steps, imbalance_reaction=False, n_samples=n_samples, level=30, volume=volume, side='ask', place_at=place)
                else:
                    SM = SampleMarket(n_steps=max_steps, imbalance_reaction=True, n_samples=n_samples, level=30, volume=volume, side='ask', place_at=place, damping_factor=d)

                start = time.time()
                out = SM.mp_sample(seed=0, n_workers=n_workers)
                # out = SM.sample(seed=0)
                out = list(itertools.chain.from_iterable(out)) 
                out = np.array(out)
                print(out.shape)
                print(f'{(time.time()-start)} seconds to sample')

                # fig, ax = plt.subplots() 
                # ax.hist(out, bins=20) 
                # if reaction:
                #     ax.set_title(f'Fill Time Distribution, {volume} lots, with imbalance reaction, place at {placed_at}')
                # else:
                #     ax.set_title(f'Fill Time Distribution, {volume} lots, without imbalance, place at {placed_at}')             
                # ax.set_xlabel('Fill Time') 
                # ax.set_ylabel('Frequency') 
                # ax.set_xlim(0, max_steps) 
                # ax.set_ylim(0, 500)     
                # plt.xlim(0, max_steps)
                # out = np.array(out)
                # print('fill probability')
                data[column_name].append(np.sum(out < max_steps-1)/len(out))
                # print(np.sum(out < max_steps-1)/len(out))
        # print(data)
        data = pd.DataFrame.from_dict(data)
        data.index = placed_at
        data.index.name = 'level'
        data.columns.name = 'damping factor'
        data = data.round(2)
        print(data)
        data.to_csv(f'results/{volume}_lots_fill_probabilities.csv')


    # plt.show()

    # plt.tight_layout()
    # plt.savefig(f'plots/fill_time_distribution_{volume}_lots_{im}_imbalance_placed_at_{place}.pdf', dpi=400)

    # if imbalance_reaction:
    #     np.savez(f'data/fill_probability_{volume}_lots_ir.npz', out=out)
    # else:
    #     np.savez(f'data/fill_probability_{volume}_lots.npz', out=out)

    # print(f'{(time.time()-start)/60} minutes to sample')
        
    # print(data)
    # data = pd.DataFrame.from_dict(data)
    # data.index = volumes
    # # data.index = placed_at
    # data.index.name = 'lots'
    # # data.index.name = 'level' 
    # data = data.round(2)

    # data.to_latex(f'tables/fill_time_distribution.tex', index=True, float_format="%.2f")
    # # only vary the levels not volume
    # data.to_latex(f'tables/fill_time_distribution_per_level.tex', index=True, float_format="%.2f")


    # print(1)

        


# plt.hist(out, bins=50)

# plt.show()

# print(out)

# for volume in volumes:
#     placed_at = 0
#     M = Market(seed=1, side=side, volume=volume, place_at=placed_at)

#     fill_times = []
#     filled = []

#     for n_samples in range(n_samples):
#         order_filled = False
#         M.reset()
#         if n_samples % 100 == 0:
#             print(n_samples)
#         for t in range(max_steps):
#             order_filled = M.step()
#             if order_filled:
#                 break 
#         fill_times.append(t)
#         if order_filled:
#             filled.append(1)
#         else:
#             filled.append(0)
        
#     print(f'average fill time: {sum(fill_times)/len(fill_times)}') 
#     print(f'fill probability: {sum(filled)/len(filled)}') 


#     if side == 'bid':
#         color = 'blue'
#     else: 
#         color = 'red'

#     ## change font size on axis 
#     plt.hist(fill_times, bins=50, color=color)
#     plt.title(f'fill time distribution for {volume} lot at {placed_at} level, max={max_steps} steps')
#     plt.fontsize = 16
#     plt.xticks(fontsize=16)
#     plt.yticks(fontsize=16)
#     plt.savefig(f'plots/ft_{volume}_lots_{placed_at}_level.pdf')

#     ## 
#     plt.xlim(0, max_steps)
#     plt.show()


    

    
    






    



















