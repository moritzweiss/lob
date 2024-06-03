import numpy as np 
from numpy.random import default_rng
import time
rng = default_rng()
market_intensity = 0.1
cancel_intensities = np.array(30*[0.1])
limit_intensities = np.array(30*[0.1])
damping_weights = np.exp(-0.1*np.arange(30))
imbalance_factor = 0.5

def basic():
    bid_volumes = np.array(30*[10])
    ask_volumes = np.array(30*[10])
    bid_cancel_intensity = cancel_intensities*bid_volumes 
    ask_cancel_intensity = cancel_intensities*ask_volumes
    L = len(bid_volumes)
    if np.sum(bid_volumes) == 0:
        weighted_bid_volumes = 0
    else:
        # idx = np.nonzero(bid_volumes)[0][0]
        # weighted_bid_volumes = np.sum(self.damping_weights[:L-idx]*bid_volumes[idx:])
        weighted_bid_volumes = np.sum(damping_weights*bid_volumes)
    if np.sum(ask_volumes) == 0:
        weighted_ask_volumes = 0
    else:
        # idx = np.nonzero(ask_volumes)[0][0]
        # weighted_ask_volumes = np.sum(self.damping_weights[:L-idx]*ask_volumes[idx:])
        weighted_ask_volumes = np.sum(damping_weights*ask_volumes)
    if (weighted_bid_volumes + weighted_ask_volumes) == 0:
        imbalance = 0
    else:
        imbalance = (weighted_bid_volumes - weighted_ask_volumes) / (weighted_bid_volumes + weighted_ask_volumes)
    if np.isnan(imbalance):
        print(imbalance)
        print(bid_volumes)
        print(ask_volumes)
        raise ValueError('imbalance is nan')
    assert -1 <= imbalance <= 1, 'imbalance must be in [-1, 1]'
    pos = imbalance_factor*max(0, imbalance)
    neg = imbalance_factor*max(0, -imbalance)
    market_buy_intensity = market_intensity*(1+pos)
    market_sell_intensity = market_intensity*(1+neg)
    bid_cancel_intensity = bid_cancel_intensity*(1+neg)
    ask_cancel_intensity = ask_cancel_intensity*(1+pos)
    bid_limit_intensities = limit_intensities.copy()
    ask_limit_intensities = limit_intensities.copy()
    bid_limit_intensities = bid_limit_intensities*(1+pos)
    ask_limit_intensities = ask_limit_intensities*(1+neg)
     
    probability = np.array([market_sell_intensity, market_buy_intensity, np.sum(bid_limit_intensities), np.sum(ask_limit_intensities), np.sum(bid_cancel_intensity), np.sum(ask_cancel_intensity)])        
    assert np.all(probability >= 0), 'all probabilities must be > 0'
    waiting_time = rng.exponential(1/np.sum(probability))

basic()


start_time = time.time()
for i in range(int(2e3)):
    basic()
end_time = time.time()

execution_time = end_time - start_time
print("Execution time:", execution_time)