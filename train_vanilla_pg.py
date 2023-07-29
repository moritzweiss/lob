from ray.rllib.algorithms.pg import PGConfig
from ray import air 
# from ray.rllib.algorithms.ppo import PPOConfig
from advanced_multi_lot import Market 
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
import ray 
from ray import tune
from ray.rllib.models import ModelCatalog
from custom_model import CustomTorchModel
import copy 
from ray.rllib.models import MODEL_DEFAULTS

# ray.init(local_mode=True)

from gymnasium.utils.env_checker import check_env

env_config = {'total_n_steps': int(1e3), 'log': False, 'seed': None, 'initial_level': 2, 'initial_volume': 10}
M = Market(config=env_config)
check_env(M)

ModelCatalog.register_custom_model("my_torch_model", CustomTorchModel)
custom_config = copy.deepcopy(MODEL_DEFAULTS)
custom_config = {}
custom_config['test'] = 1

model_config = { # By default, the MODEL_DEFAULTS dict above will be used.
        "custom_model": "my_torch_model",
         "custom_model_config": custom_config} 

config = (PGConfig().rollouts(num_rollout_workers=0, create_env_on_local_worker=False, batch_mode='complete_episodes', observation_filter='NoFilter')
        .framework('torch')
        .resources(num_gpus=0)
        .environment(env=Market, env_config=env_config)
        .training(train_batch_size=128, model=model_config, lr=1e-4)
        # .training(train_batch_size=4000, gamma=1, lr = tune.grid_search([1e-4, 1e-3]))
        .debugging(fake_sampler=False)
        ) 

algo = config.build()
algo.train()
# algo.evaluate()


# tuner = tune.Tuner(
#     "PG",
#     run_config=air.RunConfig(
#         stop={"training_iteration": 50}, storage_path = '/Users/weim/projects/lob_simulator/cont_model/results',  name = 'test', 
#         checkpoint_config=air.CheckpointConfig(checkpoint_frequency=5, checkpoint_at_end=True, checkpoint_score_order='max', 
#                                                checkpoint_score_attribute='episode_reward_mean')
#     ),
#     param_space=config,
# )

# tuner.fit()






