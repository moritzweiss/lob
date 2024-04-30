import numpy as np
# parameters from the paper 
limit_intensities = np.array([0.2842, 0.5255, 0.2971, 0.2307, 0.0826, 0.0682, 0.0631, 0.0481, 0.0462, 0.0321, 0.0178, 0.0015, 0.0001])
market_intensity = 0.1237
cancel_intensities = np.array([0.8636, 0.4635, 0.1487, 0.1096, 0.0402, 0.0341, 0.0311, 0.0237, 0.0233, 0.0178, 0.0127, 0.0012, 0.0001])
base ={}
base['intensities'] = {}
base['intensities']['limit'] = limit_intensities
base['intensities']['market'] = market_intensity
base['intensities']['cancellation'] = 1e-3*cancel_intensities
base['volumes'] = {}
base['distribution'] = 'log_normal'
base['volumes']['market'] = {'mean': 4.0, 'std': 1.19}
base['volumes']['limit'] = {'mean': 4.47, 'std': 0.83}
base['volumes']['cancellation'] = {'mean': 4.48, 'std': 0.82}
base['volumes']['clipping'] = {'min': 1, 'max': 2000}   

noise_agent_config = {}
# generated smaller queues than the base config 
# volumes are half normal with mean 1 and sigma 3, they are clipped by 1, 20
# the reasoning for the sigma is that events that eat into more levels of the book than the first two levels are unlikely 
# note: in the new set up, we might not a sigma that is so high 
# still need to investigate this 
# intensities
noise_agent_config['market_intensity'] = market_intensity
noise_agent_config['limit_intensities'] = limit_intensities
noise_agent_config['cancel_intensities'] = 1e-1*cancel_intensities
# volume related things 
noise_agent_config['volume_distribution'] = 'half_normal'
noise_agent_config['market_mean'] = 1
noise_agent_config['market_std'] = 2
noise_agent_config['limit_mean'] = 1
noise_agent_config['limit_std'] = 1
noise_agent_config['cancel_mean'] = 1
noise_agent_config['cancel_std'] = 1
noise_agent_config['volume_min'] = 1
noise_agent_config['volume_max'] = 20
# 
noise_agent_config['unit_volume'] = False
noise_agent_config['initial_shape'] = 1 
noise_agent_config['initial_shape_file'] = None 
noise_agent_config['damping_factor'] = 1.0
noise_agent_config['imbalance_reaction'] = False
noise_agent_config['level'] = 30 
noise_agent_config['imbalance_factor'] = 3
# 
noise_agent_config['rng'] = np.random.default_rng(0)
# 
noise_agent_config['initial_bid'] = 1000
noise_agent_config['initial_ask'] = 1001