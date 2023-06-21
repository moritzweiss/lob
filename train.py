from ray.rllib.algorithms import ppo
from advanced import Market 
import ray 

# not sure what happens with seeding when training 
env_config = {'total_n_steps': int(1e3), 'log': False, 'seed': None, 'initial_level': 2}

M = Market(config=env_config)

ray.rllib.utils.check_env(M)

# ray.init(local_mode=False)
# algo = ppo.PPO(env=Market, config={'env_config': config, })
# for _ in range(5):
#     result = algo.train()
#     print(result)


# from ray import tune
# Configure.
# from ray.rllib.algorithms.ppo import PPOConfig
# config = PPOConfig().environment(env=M).training(train_batch_size=4000)
# # Train via Ray Tune.
# tune.run("PPO", config=config)

# Configure.
# Train via Ray Tune.
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

config = ppo.DEFAULT_CONFIG.copy()
config['sgd_minibatch_size'] = tune.choice([256, 512])
config['clip_param'] = tune.uniform(0.25, 0.35)
config['gamma'] = 1.0
config['lr'] = tune.choice([5e-5, 5e-4])
config['env'] = Market 
config['env_config'] = env_config


tune.run("PPO", config=config, stop={"training_iteration": 50}, name='test1', local_dir='ray_results', mode='max', checkpoint_freq=2, checkpoint_at_end=True, num_samples=1)

# run analysis 
analysis = tune.ExperimentAnalysis('ray_results/test', default_metric="episode_reward_mean", default_mode="max")
best_checkpoint = analysis.best_checkpoint
df = analysis.dataframe(metric='episode_reward_mean', mode='max')
df = df.loc[:, ['episode_reward_mean', 'episode_len_mean']]

print(f'best checkpoint: {best_checkpoint}')
print('done')


