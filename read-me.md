
- create the environment python environment with anaconda using the environment.yml file 

run rl files:
- run all rl training runs with the bash script rl_files/runnner.sh 
- run individual training runs with rl_files/actor_critic.py
- actor_critic.py first trains the algorithm. then runs the algo in the environment to obtain rewards
- rewards of the trained policy are saved in the rewards folder 

run benchmark algorithms:
- run the file simulation/market_gym.py 
- there is a loop at the end of the file market_gym.py
- saves rewards to the rewards folder 

analyzing results:
- check the jupyter notebook notebooks/bar_plots_rewards.ipynb
- tables and plots were generated with this notebook 


