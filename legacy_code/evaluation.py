from ray import tune
from advanced import Market
from ray.rllib.algorithms.ppo import PPO
import time 
import numpy as np 
import ray 

env_config = {'total_n_steps': int(1e3), 'log': False, 'seed': None, 'initial_level': 2}

analysis = tune.ExperimentAnalysis('ray_results/imbalance', default_metric="episode_reward_mean", default_mode="max")

# Get the best trial
best_trial = analysis.get_best_trial("episode_reward_mean", "max")
best_checkpoint = analysis.get_best_checkpoint(best_trial).path
#get_best_checkpoint

# Load the best policy from the checkpoint
# checkpoint_path = best_trial.checkpoint.value
ray.init(local_mode=True)
checkpoint_path = best_trial.checkpoint
agent = PPO(env=Market, config={'env_config': env_config, })
agent.restore(best_checkpoint)

# # Sample the environment using the loaded policy
# state, _ = env.reset()
# done = False
# while not done:
#     action = agent.compute_single_action(state)
#     # self._get_obs(time, level), self._get_reward(reward), terminated, truncated, {}
#     state, reward, terminated, truncated, info = env.step(action)



env_config['seed'] = 0 
M = Market(config=env_config)
rewards = []
start = time.time()
for n in range(4000):
    if n%100 == 0:
        print(f'episode {n}')
    terminated = truncated = False
    observation, _ = M.reset()
    while not terminated and not truncated:
        action = agent.compute_single_action(observation)
        observation, reward, terminated, truncated, info = M.step(action)
        assert observation in M.observation_space
        if terminated or truncated:
            rewards.append(reward)

elapsed = time.time()-start
print(f'time elapsed in seconds: {elapsed}')
print(max(rewards))
print(min(rewards))
print(len(rewards))
print(np.mean(rewards))

rewards = np.array(rewards)
np.save('rewards_algo_imbalance', rewards)


x= np.load('rewards_algo_imbalance.npy')
print(x)

# evaluation results: 
# the result was -0.21
# compared to -0.252 (benchmark. submit and leave)

# for seed = 0
# for 1000 iterations we obtain 
# this was the algo with queue position and shortfall knowledge 
# algo = - 0.147
# benchmark =  -0.273


### and the algo with queue position 
# for seed = 0
# for 1000 iterations the result was -0.177
# for 2000 iterations the resilt was -0.1435, on a different run it was -0.1275, 
# yet a different run -0.1365

# benchmark is -0.269

# also try without exploration 





