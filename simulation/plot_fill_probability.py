import sys
import os
# Add parent directory to python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import numpy as np  


data = np.load('data/fill_probability_1_lots.npz')

print(data['out'].shape)

data = np.load('data/fill_probability_2_lots.npz')

print(data['out'].shape) 


data = np.load('data/fill_probability_2_lots_ir.npz')

print(data['out'].shape) 


data = np.load('data/fill_probability_1_lots_ir.npz')
print(data['out'])

print(data['out'].shape) 