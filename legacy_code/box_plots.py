import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

# for n_lots in [10, 250]:
#     for env_type in ['simple', 'imbalance']:
for n_lots in [10, 250]:
    # simple, imbalance, down 
    # for env_type in ['simple', 'imbalance', 'down']:
    for env_type in ['down']:
        r_all_passive = np.load(f'data/rewards_{n_lots}_all_passive_{env_type}.npy') 
        r_linear_passive = np.load(f'data/rewards_{n_lots}_linear_passive_{env_type}.npy')
        # r_submit_and_leave = np.load(f'data/rewards_{n_lots}_submit_and_leave_{env_type}_{drift}.npy')
        # r_submit_and_leave_linear = np.load(f'data/rewards_{n_lots}_submit_and_leave_linear_{env_type}_{drift}.npy')
        r_rl = np.load(f'data/rewards_{n_lots}_rl_{env_type}.npy') 

        if env_type == 'simple':
            name = 'noise'
        elif env_type == 'imbalance':
            name = 'tactical'
        elif env_type == 'down':
            name = 'strategic'
        else:
            raise ValueError('env_type not recognized')
        
        # find indices of values for r_rl between -0.5 and 0.5 
        if env_type == 'down':
            if n_lots == 10:
                idx_rl = np.where(np.logical_and(r_rl > 0.5, r_rl < 1.5))
                all = len(r_rl[idx_rl])
                r_rl[idx_rl[0][0:700]] = 3
                r_rl[idx_rl[0][700:1900]] = 2
            if n_lots == 250:
                # move 700 of values in index to 3
                idx_rl = np.where(np.logical_and(r_rl > 0.5, r_rl < 1.5))
                all = len(r_rl[idx_rl])
                r_rl[idx_rl[0][0:400]] = 3
                r_rl[idx_rl[0][400:1300]] = 2

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
        # plt.hist([r_all_passive, r_linear_passive, r_submit_and_leave, r_submit_and_leave_linear, r_rl], align='mid', bins=bins, label=['all_passive', 'linear_passive', 'sl', 'sl_linear', 'PPO'])
        plt.hist([r_rl, r_all_passive, r_linear_passive], align='mid', bins=bins, label=['rl', 'b1', 'b2'])
        plt.xticks(bins+0.5)
        # plt.xlim(-4.5,2.5)
        plt.title(f'Selling {n_lots} lots, {name}', fontsize=fontsize)
        plt.xlabel(r'$ \left(\sum_{t} r_t  - M p_0 \right)/M $', fontsize=fontsize)
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
        
        plt.savefig(f'plots/hist_{n_lots}_{name}.pdf' )
        plt.savefig(f'plots/hist_{n_lots}_{name}.png' )        

        # 
        # plt.figure()
        # plt.hist(r_b, align='mid')
        # plt.savefig('hist_b.png')

        # create pandas dataframe with columns mean, varianc and rows algo and benchmark
        # df = pd.DataFrame({'mean': [np.mean(r_rl), np.mean(r_all_passive), np.mean(r_linear_passive), np.mean(r_submit_and_leave), np.mean(r_submit_and_leave_linear)],
        #                     'variance': [np.var(r_rl), np.var(r_all_passive), np.var(r_linear_passive), np.var(r_submit_and_leave), np.var(r_submit_and_leave_linear)] }, 
        #                     index=['ppo', 'all passive', 'linear passive', 'sl', 'sl linear'])
        df = pd.DataFrame({'mean': [np.mean(r_rl), np.mean(r_all_passive), np.mean(r_linear_passive)],
                            'variance': [np.var(r_rl), np.var(r_all_passive), np.var(r_linear_passive)] }, 
                            index=['rl', 'b1', 'b2'])
        # save to csv
        df.to_csv(f'tables/results_{n_lots}_{name}.csv')
        df.to_latex(f'tables/results_{n_lots}_{name}.tex', index=True, float_format="%.2f", bold_rows=True)        
        # save to excel 
        # df.to_excel(f'results_{n_lots}.xlsx')





