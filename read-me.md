python setup:
- create the python environment with pip or anaconda using requirements.txt file 
- we use python version 3.9

run rl files:
- run all rl training runs with the bash script rl_files/runnner.sh 
- run individual training runs with rl_files/actor_critic.py
- actor_critic.py first trains the algorithm. then runs the algo in the environment to obtain rewards that are used for tables and density plots.
- by default, we collect 10000 rewards from the trained policy
- rewards of the trained policy are saved in the rewards folder 

run benchmark algorithms:
- run the file simulation/market_gym.py 
- there is a loop at the end of the file market_gym.py, which collects 10000 samples for all environments and lot sizes
- saves rewards to the rewards folder 

analyzing results:
- check the jupyter notebook notebooks/bar_plots_rewards.ipynb
- tables and plots were generated with this notebook 

data:
- rewards generated by the trained RL policies and benchmark algorithms are in the rewards folder 
- initial shapes, from which we start the simulation, are in the initial_shape/ folder
- tensorboard training curves in the folder runs/