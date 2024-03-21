import os, sys 
current_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_path)
sys.path.append(parent_dir)
from ray.rllib.algorithms.ppo import PPOConfig
from ray import air 
from ray import tune
import copy 
from ray.rllib.models import MODEL_DEFAULTS
from gymnasium.utils.env_checker import check_env
from simulation.all_markets_simulation import Market, config 
import sys


# ray.init(local_mode=args.local_mode)
env_config = config.copy()
M = Market(config=env_config)
check_env(M)

# custom config 
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


config = (PPOConfig().rollouts(num_rollout_workers=40, batch_mode='complete_episodes', observation_filter='NoFilter')
        .framework('torch')
        .resources(num_gpus=0)
        .environment(env=Market, env_config=config)
        # .training(train_batch_size=1024, gamma=1.0, lr = 1e-3, sgd_minibatch_size=256 , num_sgd_iter=1 , use_kl_loss=False , clip_param=tune.grid_search([0.3, 0.5, 1.0, 3.0]) , vf_clip_param=100.0, use_gae=False, use_critic=True, vf_loss_coeff=1.0, model=model_config)
        .training(train_batch_size=1024, gamma=1.0, lr = 1e-3, sgd_minibatch_size=256 , num_sgd_iter=2 , use_kl_loss=False , clip_param=tune.grid_search([1.0]) , vf_clip_param=100.0, use_gae=False, use_critic=True, vf_loss_coeff=1.0, model=model_config)
        .debugging(fake_sampler=False)
        .environment(disable_env_checking=True)
        .reporting(metrics_num_episodes_for_smoothing=500)
        # .rl_module(_enable_rl_module_api=False)
        ) 


tuner = tune.Tuner(
    "PPO",
    run_config=air.RunConfig(
        stop={"training_iteration": 100}, storage_path = f"{parent_dir}/ray_results",  name = f"{env_config['volume']}_{env_config['type']}", 
        checkpoint_config=air.CheckpointConfig(checkpoint_frequency=5, checkpoint_at_end=True, checkpoint_score_order='max', 
                                               checkpoint_score_attribute='episode_reward_mean'), verbose=1,
    ),
    param_space=config,
)


results = tuner.fit()
