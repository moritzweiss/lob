import numpy as np 
import matplotlib.pyplot as plt 

r_a = np.load('rewards_algo_imbalance.npy') + 1
r_b = np.load('rewards_bench.npy') + 1 

print(f'mean and variance of benchmark') 
print(f'mean: {np.mean(r_b)}, var: {np.var(r_b)}')
print('mean and variance of algo')
print(f'mean: {np.mean(r_a)}, var: {np.var(r_a)}')


bins = np.arange(-4,  3.5) - 0.5

fontsize = 14

# plt.boxplot([r_a, r_b])
plt.figure()
ax = plt.gca()
plt.grid(axis='y')
ax.set_axisbelow(True)
plt.hist([r_b, r_a], align='mid', bins=bins, label=['Benchmark', 'PPO'])
plt.xticks(bins+0.5)
plt.xlim(-4.5,2.5)
plt.xlabel(r'$ \left(\sum_{t} r_t \right) - p_0^b$', fontsize=fontsize)
plt.ylabel('episode count', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

handles, labels = plt.gca().get_legend_handles_labels()
order = [1, 0]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=fontsize)
plt.tight_layout()
plt.savefig('hist.pdf' )

# 
# plt.figure()
# plt.hist(r_b, align='mid')
# plt.savefig('hist_b.png')


print(1)



