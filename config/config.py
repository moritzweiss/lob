import numpy as np

base ={}


limit_intensities = np.array([0.2842, 0.5255, 0.2971, 0.2307, 0.0826, 0.0682, 0.0631, 0.0481, 0.0462, 0.0321, 0.0178, 0.0015, 0.0001])
market_intensity = 0.1237
cancel_intensities = np.array([0.8636, 0.4635, 0.1487, 0.1096, 0.0402, 0.0341, 0.0311, 0.0237, 0.0233, 0.0178, 0.0127, 0.0012, 0.0001])



## base case from the paper 


# intensities
base['intensities'] = {}

base['intensities']['limit'] = limit_intensities

base['intensities']['market'] = market_intensity

base['intensities']['cancellation'] = 1e-3*cancel_intensities


# volumes 
base['volumes'] = {}

base['distribution'] = 'log_normal'

base['volumes']['market'] = {'mean': 4.0, 'std': 1.19}

base['volumes']['limit'] = {'mean': 4.47, 'std': 0.83}

base['volumes']['cancellation'] = {'mean': 4.48, 'std': 0.82}

base['volumes']['clipping'] = {'min': 1, 'max': 2000}   

# base['level'] = 30 



## new config  

config1 = {}

# config1['level']  = base['level']

# generated smaller queues than the base config 
# volumes are half normal with mean 1 and sigma 3, they are clipped by 1, 20
# the reasoning for the sigma is that events that eat into more levels of the book than the first two levels are unlikely 

# intensities
config1['intensities'] = {}
config1['intensities']['market'] = market_intensity
config1['intensities']['limit'] = limit_intensities
config1['intensities']['cancellation'] = 1e-1*cancel_intensities



# volume distributions

config1['distribution'] = 'half_normal_plus1'

config1['volumes'] = {}

config1['volumes']['market'] = {'mean': 0.0, 'std': 3.0}

config1['volumes']['limit'] = {'mean': 0.0, 'std': 3.0}

config1['volumes']['cancellation'] = {'mean': 0.0, 'std': 3.0}

config1['volumes']['clipping'] = {'min': 1, 'max': 20}   


##

config = {0: base, 1: config1}
