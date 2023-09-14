import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

for n_lots in [10, 250]:
    for env_type in ['simple', 'imbalance']:

        r_all_passive = np.load(f'data/rewards_{n_lots}_all_passive_{env_type}.npy') 
        r_linear_passive = np.load(f'data/rewards_{n_lots}_linear_passive_{env_type}.npy')
        r_rl = np.load(f'data/rewards_{n_lots}_rl_{env_type}.npy') 

        # print(r_a)
        # print('######')
        # print(f'mean and variance of benchmark') 
        # print(f'mean: {np.mean(r_b)}, var: {np.var(r_b)}')
        # print(f'min of benchmark: {np.min(r_b)}')
        # print(f'max of benchmark: {np.max(r_b)}')
        # print('######')
        # print('mean and variance of algo')
        # print(f'mean: {np.mean(r_a)}, var: {np.var(r_a)}')
        # print(f'min of algo: {np.min(r_a)}')
        # print(f'max of algo: {np.max(r_a)}')

        # figure 
        # plt.figure() 
        # bins = np.linspace(-5, 2, 50)
        # plt.hist(r_b, bins=bins, alpha=0.5, label='Benchmark')
        # plt.hist(r_a, bins=bins, alpha=0.5, label='PPO')
        # plt.legend(loc='upper right')

        # plt.set_cmap('Dark2') 

        # import matplotlib as mpl
        # mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["r", "k", "c"]) 

        bins = np.arange(-5, 5) - 0.5

        fontsize = 14

        # plt.boxplot([r_a, r_b])
        plt.figure()
        ax = plt.gca()
        plt.grid(axis='y')
        ax.set_axisbelow(True)
        plt.hist([r_all_passive, r_linear_passive, r_rl], align='mid', bins=bins, label=['all_passive', 'linear_passive', 'PPO'])
        plt.xticks(bins+0.5)
        # plt.xlim(-4.5,2.5)
        plt.title(f'Selling {n_lots} lots, {env_type}', fontsize=fontsize)
        plt.xlabel(r'$ \left(\sum_{t} r_t  - V_0 p_0^b \right)/V_0$', fontsize=fontsize)
        plt.ylabel('episode count', fontsize=fontsize)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=fontsize)
        plt.ylim(0, 4500)

        handles, labels = plt.gca().get_legend_handles_labels()
        # order = [1, 0]
        # plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=fontsize)
        # add legend with location 
        plt.legend(fontsize=fontsize, loc='upper left')
        plt.tight_layout()
        plt.savefig(f'plots/hist_{n_lots}_{env_type}.pdf' )
        plt.savefig(f'plots/hist_{n_lots}_{env_type}.png' )

        # 
        # plt.figure()
        # plt.hist(r_b, align='mid')
        # plt.savefig('hist_b.png')


        # create pandas dataframe with columns mean, varianc and rows algo and benchmark
        df = pd.DataFrame({'mean': [np.mean(r_rl), np.mean(r_all_passive), np.mean(r_linear_passive)], 'variance': [np.var(r_rl), np.var(r_all_passive), np.var(r_linear_passive)]}, index=['ppo', 'all passive', 'linear passive'])
        # save to csv
        df.to_csv(f'tables/results_{n_lots}_{env_type}.csv')
        df.to_latex(f'tables/results_{n_lots}_{env_type}.tex', index=True, float_format="%.2f", bold_rows=True)
        # save to excel 
        # df.to_excel(f'results_{n_lots}.xlsx')





