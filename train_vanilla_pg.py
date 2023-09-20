# from ray.rllib.algorithms.ppo import PPOConfig
# from ray.rllib.algorithms.pg import PGConfig
# from ray.rllib.algorithms.sac.sac import SACConfig
# from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
# from ray.rllib.utils.spaces.simplex import Simplex
# from ray.rllib.algorithms.a2c.a2c import A2CConfig
# import ray 
import os, sys 
current = os.path.dirname(os.path.realpath(__file__))
# parent = os.path.dirname(current)
# sys.path.append(parent)
from ray.rllib.algorithms.ppo import PPOConfig
from ray import air 
from advanced_multi_lot import Market 
from ray import tune
# from ray.rllib.models import ModelCatalog
# from custom_model import CustomTorchModel
# from custom_model import TorchDirichlet
import copy 
from ray.rllib.models import MODEL_DEFAULTS
from gymnasium.utils.env_checker import check_env

# naming: 250_imbalace, 250_simple, (#lots_imbalance, #lots_simple)
# same for 100 lots 

# n_lots 10 or 250
# env_type: imbalance or simple

# from ray.rllib.models.torch.torch_action_dist import TorchDirichlet
# ray.init(local_mode=True)


env_config = {'total_n_steps': int(1e3), 'log': False, 'seed': None, 'initial_level': 2, 'initial_volume': 10, 'env_type': 'down'}
M = Market(config=env_config)
check_env(M)

# this is needed for new environment configurations 

# this stuff is needed for custom models 
# ModelCatalog.register_custom_model("my_torch_model", CustomTorchModel)
# ModelCatalog.register_custom_action_dist("my_dist", TorchDirichlet)

# model config: switch of shared layers 
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


# example: learning rate scheduler 
# config = PPOConfig().training(lr=[[int(0), 1e-3], [int(1e4), 1e-4], [int(1e5), 5e-5], [int(2e5), 1e-5], [int(5e5), 5e-6]])
# learning rate scheduler does not seem to be working 

# bunch of configurations 
config = (PPOConfig().rollouts(num_rollout_workers=17, batch_mode='complete_episodes', observation_filter='NoFilter')
        .framework('torch')
        .resources(num_gpus=0)
        .environment(env=Market, env_config=env_config)
        # .training(train_batch_size=1024, optimization_config={
        #     "actor_learning_rate": 1e-3,
        #     "critic_learning_rate": 1e-3,
        #     "entropy_learning_rate": 1e-3,
        # })
        # .training(train_batch_size=1024, lr=tune.grid_search([1e-3, 5e-4, 1e-4, 5e-5]), _enable_learner_api=False)
        # .training(train_batch_size=128, model=model_config, lr=1e-3, _enable_learner_api=False)
        # .training(train_batch_size=8000, gamma=1.0, lr = tune.grid_search([5e-5, 1e-5]), model=model_config, _enable_learner_api=False, sgd_minibatch_size=2048, num_sgd_iter=8, clip_param=0.4)
        # .training(train_batch_size=8000, gamma=1.0, lr = tune.grid_search([5e-5, 1e-5]), model=model_config, _enable_learner_api=False, sgd_minibatch_size=2048, num_sgd_iter=8, clip_param=0.4)
        # .training(train_batch_size=8000, gamma=1.0, lr = tune.grid_search([5e-5, 1e-5]), model=model_config, _enable_learner_api=False, sgd_minibatch_size=2048, num_sgd_iter=8, clip_param=0.4)
        # .training(train_batch_size=8192, gamma=1.0, lr = tune.grid_search([1e-3, 5e-4]), model=model_config, _enable_learner_api=False)
        # .training(train_batch_size=8192, gamma=1.0, model=model_config, _enable_learner_api=False, lr_schedule=[[int(0), 1e-3], [int(2e5), 1e-4], [int(3e5), 5e-5], [int(4e5), 1e-5]])
        # .training(train_batch_size=8192, gamma=1.0, model=model_config, _enable_learner_api=False, lr_schedule=[[int(0), 1e-2], [int(1e5), 1e-3],[int(2e5), 5e-4]])
        # .training(train_batch_size=8192, gamma=1.0, model=model_config, _enable_learner_api=False, lr= tune.grid_search([1e-3, 1e-4, 5e-5]))
        # .training(train_batch_size=8192, gamma=1.0, model=model_config, _enable_learner_api=False, lr= tune.grid_search([1e-2, 1e-3, 1e-4]))
        # .training(train_batch_size=233, gamma=1.0, model=model_config, _enable_learner_api=False, lr= 1e-5)
        # .training(train_batch_size=8192, gamma=1.0, lr=1e-3, model=model_config, _enable_learner_api=False, sgd_minibatch_size=2048)
        # .training(train_batch_size=8192, gamma=1.0, lr = 1e-2,  model=model_config, _enable_learner_api=False, sgd_minibatch_size=4096, num_sgd_iter=tune.grid_search([2,2,4,4]), use_kl_loss=False , clip_param=10.0, vf_clip_param=100.0, use_gae=False, use_critic=True)
        # .training(train_batch_size=4096, gamma=1.0, lr = 1e-2,  model=model_config, _enable_learner_api=False, sgd_minibatch_size=4096, num_sgd_iter=tune.grid_search([1, 2]), use_kl_loss=False , clip_param=10.0, vf_clip_param=100.0, use_gae=False, use_critic=True)
        # .training(train_batch_size=4096, gamma=1.0, lr = 1e-2,  model=model_config, _enable_learner_api=False, sgd_minibatch_size=2048, num_sgd_iter=tune.grid_search([1, 1, 4, 4]), use_kl_loss=False , clip_param=10.0, vf_clip_param=100.0, use_gae=False, use_critic=True)
        # .training(train_batch_size=8192, gamma=1.0, lr = 1e-2,  model=model_config, _enable_learner_api=False, sgd_minibatch_size=1024 , num_sgd_iter=1 , use_kl_loss=False , clip_param=tune.grid_search([0.3, 0.3, 0.5, 0.5]), vf_clip_param=100.0, use_gae=False, use_critic=True)
        # .training(train_batch_size=8192, gamma=1.0, lr = 1e-2,  model=model_config, _enable_learner_api=False, sgd_minibatch_size=tune.grid_search(4*[1024]) , num_sgd_iter=1 , use_kl_loss=False , clip_param=0.3, vf_clip_param=100.0, use_gae=False, use_critic=True, vf_loss_coeff=1.0)
        # .training(train_batch_size=8192, gamma=1.0, lr = 1e-3 , _enable_learner_api=False, sgd_minibatch_size=1024 , num_sgd_iter=1 , use_kl_loss=False , clip_param=0.3 , vf_clip_param=100.0, use_gae=False, use_critic=True, vf_loss_coeff=1.0,) #  model=model_config)
        .training(train_batch_size=8192, gamma=1.0, lr = 1e-3, _enable_learner_api=False, sgd_minibatch_size=1024 , num_sgd_iter=1 , use_kl_loss=False , clip_param=tune.grid_search([0.3, 0.5, 1.0, 3.0]) , vf_clip_param=100.0, use_gae=False, use_critic=True, vf_loss_coeff=1.0, model=model_config)
        # .training(train_batch_size=512, gamma=1.0, lr = 1e-3, _enable_learner_api=False, sgd_minibatch_size=128 , num_sgd_iter=1 , use_kl_loss=False , clip_param=0.3 , vf_clip_param=100.0, use_gae=False, use_critic=True, vf_loss_coeff=1.0, model=model_config)
        # .training(train_batch_size=16384, gamma=1.0, lr = 1e-2,  model=model_config, _enable_learner_api=False, sgd_minibatch_size=tune.grid_search([1024, 2048]) , num_sgd_iter=tune.grid_search([1,2]) , use_kl_loss=False , clip_param=0.4, vf_clip_param=100.0, use_gae=False, use_critic=True)
        # .training(gamma=1.0, lr = tune.grid_search([1e-2, 1e-2, 5e-3, 1e-3]),  model=model_config, _enable_learner_api=False, train_batch_size=8192)
        # .training(train_batch_size=8192, gamma=1.0, lr = tune.grid_search([1e-2, 1e-2, 1e-2, 1e-2]), model=model_config, _e1nable_learner_api=False)
        # .training(train_batch_size=8192, gamma=1.0, lr = tune.grid_search([1e-2, 1e-2, 1e-2, 1e-2]), model=model_config, _enable_learner_api=False)
        # .training(train_batch_size=128, gamma=1.0, lr = 1e-2, model=model_config, _enable_learner_api=False, sgd_minibatch_size=128, num_sgd_iter=1, use_kl_loss=False, clip_param=2.0, vf_clip_param=200, use_gae=False, use_critic=True)
        .debugging(fake_sampler=False)
        .environment(disable_env_checking=False)
        .rl_module(_enable_rl_module_api=False)
        .reporting(metrics_num_episodes_for_smoothing=2000)
        # .evaluation(evaluation_interval=5, evaluation_duration=4000, evaluation_num_workers=10, evaluation_duration_unit='episodes')
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
        stop={"training_iteration": 50}, storage_path = f"{current}/results",  name = f"{env_config['initial_volume']}_{env_config['env_type']}", 
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







