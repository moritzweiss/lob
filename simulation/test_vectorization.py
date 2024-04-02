
from all_markets_simulation import Market, config 
import gymnasium as gym 
from multiprocessing import Pool
import numpy as np
import itertools
import time 
import ray

# config['execution_agent'] = 'sl_agent'
# config['execution_agent'] = 'linear_sl_agent'
config['execution_agent'] = 'sl_agent'
# print(config)

# Question? how is seeding handled by rllib rollouts ???
# even if all parallel environemnts are seeded the same, i guess it should still be plenty of variation
# probably doesnt matter so much 

def _make_env(seed):    
    cf = config.copy()
    cf['seed'] = seed
    # print(config)
    # env = Market(config=config)
    return lambda: Market(config=cf)

print('#####')


# @ray.remote
def rollout(seed, num_episodes, strategy = 'sl_agent', env_type='flow', damping_factor=0.5, volume=50):
    c = config.copy()
    c['seed'] = seed
    c['execution_agent'] = strategy
    c['type'] = env_type
    c['volume'] = volume
    c['damping_factor'] = damping_factor
    # print(c)
    env = Market(config=c)
    total_rewards = []
    for _ in range(num_episodes):
        observation = env.reset()
        total_reward = 0
        terminated = False
        while not terminated:
            action = env.action_space.sample()  # Sample a random action
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        total_rewards.append(total_reward)
    return total_rewards


def mp_rollout(n_samples, n_envs, strategy, env_type, damping_factor, volume):
    samples_per_env = int(n_samples/n_envs)
    # print(f'starting {n_envs} workers')
    # print(f'{samples_per_env} per worker')
    with Pool(n_envs) as p:
        all_rewards = p.starmap(rollout, [(seed, samples_per_env, strategy, env_type, damping_factor, volume) for seed in range(n_envs)])    
    all_rewards = list(itertools.chain.from_iterable(all_rewards))
    return all_rewards


if __name__ == '__main__':

    # rewards = rollout(0, 10, 'sl_agent', 'flow', 0.1)

    # assert [lambda: Market(set_seed(0)), lambda: Market(set_seed(1))] == [lambda: Market(set_seed(seed)) for seed in [0,1]]
    # n_ens = 2

    n_samples = 1000
    n_envs = 50

    # start = time.time()
    # out = [rollout.remote(seed, samples_per_env) for seed in range(n_envs)]
    # out = ray.get(out)
    # out = list(itertools.chain.from_iterable(out))
    # # print(ray.get(out))
    # print('time', time.time() - start)
    # print(f'length: {len(out)}')
    # out = rollout(0, 100)
    # # print(out) 
    

    # env_functions = [lambda: Market(config), lambda: Market(config)]
    # envs = gym.vector.AsyncVectorEnv([lambda: Market(set_seed(0)), lambda: Market(set_seed(2))])
    # envs = gym.vector.AsyncVectorEnv(env_functions)

    # n_envs = 1 
    start = time.time()
    all_rewards = mp_rollout(n_samples, n_envs, 'sl_agent', 'flow', 1.0, 40)
    print('time', time.time() - start)
    print(f'length: {len(all_rewards)}')
    print(f'mean reward: {np.mean(all_rewards)}')
    print(f'std reward: {np.std(all_rewards)}')             
    print('done')

