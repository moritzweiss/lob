import matplotlib.pyplot as plt
import pandas as pd
import itertools
import time 
from average_shape_vectorized import NoiseAgent
import gymnasium as gym 
from limit_order_book.limit_order_book import LimitOrderBook
from limit_order_book.limit_order_book import LimitOrder
from limit_order_book.plotting import heat_map
from multiprocessing import Pool
from numpy.random import default_rng
import numpy as np
import sys
import os
# import mat
# Add parent directory to python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# ToDo: write a test for limit order fills by agent: place order somewhere in the book and check if it is filled
# make fill messages consistent: the same for every order type ! 
# ToDo: write a test for fill time of limit orders in a deterministic setting 


class Market(gym.Env):
    def __init__(self, rng, n_steps=int(1e3), config_n=1, n_samples=10, level=30, volume=10, side='ask', imbalance_reaction=False, place_at=1, initial_shape_file='data_small_queue.npz') -> None:
        self.noise_agent = NoiseAgent(level=level, rng=rng, config_n=config_n, imbalance_reaction=imbalance_reaction, initial_shape_file=initial_shape_file, damping_factor=0.0)
        self.level = level
        self.n_steps = n_steps
        self.volume = volume 
        self.side = side
        self.place_at = place_at        
        # self.n_samples = n_samples
        return None 
    
    def reset(self):
        self.lob = LimitOrderBook(level=self.level, list_of_agents=['noise_agent', 'smart_agent', 'strategic_agent'])
        self.time = 0
        # initialize the book 
        orders = self.noise_agent.initialize()
        [self.lob.process_order(order) for order in orders]
        # placement by strategic investor  
        # 100 noise transitions 
        for _ in range(100):
            order = self.noise_agent.generate_order(self.lob.data.best_bid_prices[-1], self.lob.data.best_ask_prices[-1], self.lob.data.bid_volumes[-1], self.lob.data.ask_volumes[-1])
            self.lob.process_order(order)
            self.time += 1
        if self.side == 'ask':
            price = self.lob.get_best_price('ask')
            order = LimitOrder('smart_agent', 'ask', price+self.place_at, self.volume)
        else:
            price = self.lob.get_best_price('bid')
            order = LimitOrder('smart_agent', 'bid', price, self.volume)
        self.lob.process_order(order) 
        # self.time = 0 
        return None
    
    def step(self):
        terminated = False
        order = self.noise_agent.generate_order(self.lob.data.best_bid_prices[-1], self.lob.data.best_ask_prices[-1], self.lob.data.bid_volumes[-1], self.lob.data.ask_volumes[-1])
        self.lob.process_order(order)
        # print(order.type)
        if order.type == 'market':
            if not self.lob.order_map_by_agent['smart_agent']:
                terminated = True
                return terminated
                # self.lob.process_order(out[0])
        self.time += 1
        return terminated
    


class SampleMarket:
    def __init__(self, n_steps=int(1e3), config_n=1, n_samples=10, level=30, volume=10, side='ask', imbalance_reaction=False, place_at=1, initial_shape_file='data_small_queue.npz'):
        # self.noise_agent = NoiseAgent(level=level, rng=rng, config_n=config_n, imbalance_reaction=imbalance_reaction, initial_shape_file=initial_shape_file)
        self.level = level
        self.n_steps = n_steps
        self.volume = volume 
        self.side = side
        self.place_at = place_at
        self.n_samples = n_samples
        self.imbalance_reaction = imbalance_reaction
        return None
    def sample(self, seed):
        M = Market(rng=default_rng(seed), imbalance_reaction=self.imbalance_reaction,  n_steps=self.n_steps, n_samples=self.n_samples, level=self.level, volume=self.volume, side=self.side, place_at=self.place_at)
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

    def mp_sample(self, n_workers=2, seed=0):
        seeds = [i for i in range(n_workers)]
        # seeds = n_workers*[seed]
        p = Pool(n_workers)
        results = p.map(self.sample, seeds)
        p.close()
        # p.terminate()
        # results
        return results

 
if __name__ == '__main__':

    n_workers = 50
    n_samples = int(20)
    max_steps = int(1e3)
    # volumes = [1, 2]
    volumes = [40]
    # volumes = [1]
    # volumes = [1]
    # placed_at = [0,1,2,3]
    # placed_at = [0,1,2,3]
    placed_at = [0,1,2]
    reaction = [False, True]
    # reaction = [True]
    # reaction = [False]


    # SM = SampleMarket(n_steps=max_steps, imbalance_reaction=True, n_samples=n_samples, level=30, volume=10, side='ask', place_at=0)
    # fill_times = SM.sample(1)


    data = {}
    for imbalance_reaction in reaction:
        # for labelling 
        if imbalance_reaction:
            im = 'with'
        else:
            im = 'without'
        # only vary the levels not volume 
        # data[f'{im}_imbalance'] = []
        for place in placed_at:
            data[f'{im}_imbalance_{place}_level'] = []
            for volume in  volumes: 
                print(f'imbalance reaction: {imbalance_reaction}')
                print(f'volume: {volume}')
                print(f'place at: {place}')
                
                # loop over imbalance reaction and volume 
                SM = SampleMarket(n_steps=max_steps, imbalance_reaction=imbalance_reaction, n_samples=n_samples, level=30, volume=volume, side='ask', place_at=place)

                start = time.time()
                out = SM.mp_sample(n_workers=n_workers, seed=1)

                out = list(itertools.chain.from_iterable(out)) 

                out = np.array(out)

                # np.sum(out < max_steps)/len(out)

                data[f'{im}_imbalance_{place}_level'].append(np.sum(out < max_steps-1)/len(out))
                # data[f'{im}_imbalance'].append(np.sum(out < max_steps-1)/len(out))
                # print(data)
                # data['mean'].append(np.mean(out)) 
                # data['std'].append(np.std(out)) 

                fig, ax = plt.subplots() 
                ax.hist(out, bins=50) 
                ax.set_title(f'Fill Time Distribution, {volume} lots, {im} imbalance, place at {place}')             
                ax.set_xlabel('Fill Time') 
                ax.set_ylabel('Frequency') 
                ax.set_xlim(0, max_steps) 
                ax.set_ylim(0, 500)     
                # plt.xlim(0, max_steps)
                # plt.show()

                plt.tight_layout()
                plt.savefig(f'plots/fill_time_distribution_{volume}_lots_{im}_imbalance_placed_at_{place}.pdf', dpi=400)

                # if imbalance_reaction:
                #     np.savez(f'data/fill_probability_{volume}_lots_ir.npz', out=out)
                # else:
                #     np.savez(f'data/fill_probability_{volume}_lots.npz', out=out)

                # print(f'{(time.time()-start)/60} minutes to sample')
        
    # print(data)
    data = pd.DataFrame.from_dict(data)
    data.index = volumes
    # data.index = placed_at
    data.index.name = 'lots'
    # data.index.name = 'level' 
    data = data.round(2)
    print(data)
    data.to_csv(f'./results/fill_time_distribution.csv')
    # data.to_latex(f'tables/fill_time_distribution.tex', index=True, float_format="%.2f")
    # only vary the levels not volume
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


    

    
    






    









