import numpy as np 
import os 
import sys 
import pandas as pd
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)
import matplotlib.pyplot as plt
import seaborn as sns



n_lots = 10
env_type = 'strategic'


path = f'{parent_dir}/rewards/{env_type}_{n_lots}.csv'

data = pd.read_csv(path, index_col=0)
data.columns = ['BM2', 'BM1', 'RL']  # replace with your actual column names
# we have the order like that, because the order will be reversed by seaborn hist plot 



plt.figure(figsize=(10, 6))
# Plotting individual distributions
# sns.histplot(data, kde=True, stat='probability', legend=True, multiple='dodge', binwidth=1, binrange=(-int(np.min(l_sl)), 7), shrink=0.8)
# strtategic setting 40 lots 
if env_type == 'strategic':
    # sns.histplot(data, kde=True, stat='percent', legend=True, binwidth=2, binrange=(-18, 16), multiple='dodge', shrink=0.8, common_norm=False)
    sns.kdeplot(data, legend=True, common_norm=False, linewidth=3.5)
    plt.xticks(np.arange(-20, 20, 2))
    plt.xlim(-18, 16)
    plt.ylim(0,0.17)
elif env_type == 'flow':
    # sns.histplot(data, kde=True, stat='percent', legend=True, binwidth=2, binrange=(-18, 16), multiple='dodge', shrink=0.8, common_norm=False)
    if n_lots == 10:
        # sns.histplot(data, kde=True, stat='percent', legend=True, binwidth=1, binrange=(-8, 7), multiple='dodge', shrink=0.8, common_norm=False)
        sns.kdeplot(data, legend=True, common_norm=False, linewidth=3.5)
        plt.xticks(np.arange(-20, 20, 1))
        plt.xlim(-8, 7)
    elif n_lots == 40:
        # sns.histplot(data, kde=True, stat='percent', legend=True, binwidth=1, binrange=(-12, 5), multiple='dodge', shrink=0.8, common_norm=False)
        sns.kdeplot(data, legend=True, common_norm=False, linewidth=3.5)
        plt.xticks(np.arange(-20, 20, 1))
        plt.xlim(-12, 5)
    # sns.histplot(data, kde=True, stat='percent', legend=True, binwidth=2, binrange=(-13, 16), multiple='dodge', shrink=0.8, common_norm=False)
elif env_type == 'noise':
    if n_lots == 10:
        # sns.histplot(data, kde=True, stat='percent', legend=True, multiple='dodge', binwidth=1, binrange=(-9,6), shrink=0.8, common_norm=False)
        sns.kdeplot(data, legend=True, common_norm=False, linewidth=3.5)
        plt.xticks(np.arange(-20, 20, 1))
        plt.xlim(-9, 6)
    elif n_lots == 40:
        # sns.histplot(data, kde=True, stat='percent', legend=True, multiple='dodge', binwidth=1, binrange=(-12,7), shrink=0.8, common_norm=False)
        sns.kdeplot(data, legend=True, common_norm=False, linewidth=3.5)
        plt.xticks(np.arange(-20, 20, 1))
        plt.xlim(-12, 7)
    # plt.xticks(np.arange(-20, 20, 1))
    # plt.xlim(-12, 5)        

plt.legend(data.columns[::-1])
# plt.grid(axis='y', linestyle='-', linewidth=0.5)
plt.grid()
plt.xlabel('X', fontsize=20)
plt.ylabel('Density', fontsize=20)
# plt.ylabel('%', fontsize=18)
plt.legend(data.columns[::-1], fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig(f'histograms/{env_type}_{n_lots}.pdf')

# sns.histplot(sl, kde=True, color='blue', label='S&L')
# sns.histplot(l_sl, kde=True, color='green', label='L S&L')
# sns.histplot(rewards, kde=True, color='red', label='RL')
# plt.legend()
# plt.grid(True)
# plt.xticks(np.arange(-10, 11, 1))
# plt.xlim(-10, 7)
# plt.ylim(0, 200)
# plt.savefig('rl_sl_plots.pdf')



# Convert rewards to DataFrame for easier plotting
# rewards_df = pd.DataFrame(rewards, columns=['Rewards'])
# # Create a boxplot
# plt.figure(figsize=(10, 6))
# sns.boxplot(data=rewards_df)
# plt.title('Boxplot of Rewards')
# plt.savefig('boxplot_rewards_0.pdf')


## Create a histogram of rewards
# plt.figure(figsize=(10, 6))
# bins = np.arange(-10, 10, 0.25)
# sns.histplot(data=rewards_df, x='Rewards', bins=bins, kde=True)
# # sns.histplot(data=rewards_df, x='Rewards', bins=30, kde=True)
# plt.title('Histogram of Rewards')
# plt.savefig('histogram_rewards_0.pdf')


