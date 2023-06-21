from ray.rllib.algorithms.pg import PGConfig
from ray.rllib.algorithms.pg import DEFAULT_CONFIG 
from ray.rllib.algorithms import pg 
from ray import tune
from advanced import Market
import ray 

env_config = {'total_n_steps': int(1e3), 'log': False, 'seed': None, 'initial_level': 2}

ray.init(local_mode=False)
algo = pg.PG(env=Market, config={'env_config': env_config, })
for _ in range(5):
    result = algo.train()
    print(result)


env_config = {'total_n_steps': int(1e3), 'log': False, 'seed': None, 'initial_level': 2}
M = Market(config=env_config)
ray.rllib.utils.check_env(M)


config = DEFAULT_CONFIG.copy()
config['lr']=tune.grid_search([0.01, 0.001, 0.0001])
config['env'] = Market
config['gamma'] = 1
config['env_config'] = env_config

# config = config.environment(env=Market, env_config={'total_n_steps': int(1e3), 'log': False, 'seed': None, 'initial_level': 2})
# config = config.env_config(config={'total_n_steps': int(1e3), 'log': False, 'seed': None, 'initial_level': 2})


tune.run("PG", config=config, stop={"training_iteration": 200}, name='test_pg', local_dir='ray_results', mode='max', checkpoint_freq=2, checkpoint_at_end=True, num_samples=2)
