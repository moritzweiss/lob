import os, sys 
current_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_path)
from ray.rllib.algorithms.ppo import PPOConfig
from ray import air 
from advanced_multi_lot import Market 
from ray import tune
import copy 
from ray.rllib.models import MODEL_DEFAULTS
from gymnasium.utils.env_checker import check_env


env_config = {'total_n_steps': int(1e3), 'log': False, 'seed': None, 'initial_level': 2, 'initial_volume': 10, 'env_type': 'down'}
M = Market(config=env_config)
check_env(M)

custom_config = copy.deepcopy(MODEL_DEFAULTS)
custom_config = {}
custom_config['test'] = 1
custom_config["vf_share_layers"] = False

# this is for model configurations 
model_config = { # By default, the MODEL_DEFAULTS dict above will be used.
        "custom_model": "my_torch_model",
         "custom_model_config": custom_config,
         "custom_action_dist": "my_dist"} 
model_config = {"vf_share_layers": False} 


config = (PPOConfig().rollouts(num_rollout_workers=17, batch_mode='complete_episodes', observation_filter='NoFilter')
        .framework('torch')
        .resources(num_gpus=0)
        .environment(env=Market, env_config=env_config)
        .training(train_batch_size=1024, gamma=1.0, lr = 1e-3, _enable_learner_api=False, sgd_minibatch_size=256 , num_sgd_iter=1 , use_kl_loss=False , clip_param=tune.grid_search([0.3, 0.5, 1.0, 3.0]) , vf_clip_param=100.0, use_gae=False, use_critic=True, vf_loss_coeff=1.0, model=model_config)
        .debugging(fake_sampler=False)
        .environment(disable_env_checking=False)
        .rl_module(_enable_rl_module_api=False)
        .reporting(metrics_num_episodes_for_smoothing=500)
        ) 

# algo = config.build()
# algo.train()
# name = f"{env_config['initial_volume']}_{env_config['env_type']}"
# path = f"{current}/results"
# verbose = 0 means silent 
# verbose = 1  means status updates

tuner = tune.Tuner(
    "PPO",
    run_config=air.RunConfig(
        stop={"training_iteration": 50}, storage_path = f"{current_path}/results",  name = f"{env_config['initial_volume']}_{env_config['env_type']}", 
        checkpoint_config=air.CheckpointConfig(checkpoint_frequency=5, checkpoint_at_end=True, checkpoint_score_order='max', 
                                               checkpoint_score_attribute='episode_reward_mean'), verbose=1,
    ),
    param_space=config,
)


results = tuner.fit()



# Get the best result based on a particular metric.
# best_result = results.get_best_result(metric="episode_reward_mean", mode="max", scope='all')

# Get the best checkpoint corresponding to the best result.
# best_checkpoint = best_result.checkpoint

# from ray.rllib.algorithms.algorithm import Algorithm
# algo = Algorithm.from_checkpoint(best_checkpoint.path)







