
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
def rollout(seed, num_episodes):
    c = config.copy()
    c['seed'] = seed
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


env_functions = [_make_env(seed) for seed in range(2)]

if __name__ == '__main__':

    # assert [lambda: Market(set_seed(0)), lambda: Market(set_seed(1))] == [lambda: Market(set_seed(seed)) for seed in [0,1]]
    # n_ens = 2

    # start = time.time()
    # out = [rollout.remote(seed, 100) for seed in range(8)]
    # print(ray.get(out))
    # print('time', time.time() - start)


    # out = rollout(0, 100)
    # # print(out) 
    # print(f'length: {len(out)}')
    

    # env_functions = [lambda: Market(config), lambda: Market(config)]
    # envs = gym.vector.AsyncVectorEnv([lambda: Market(set_seed(0)), lambda: Market(set_seed(2))])
    # envs = gym.vector.AsyncVectorEnv(env_functions)

    start = time.time()
    n_envs = 8
    with Pool(n_envs) as p:
        # Map the environment creation functions to the pool of processes to run rollouts in parallel
        all_rewards = p.starmap(rollout, [(seed, 100) for seed in range(n_envs)])    
    l = list(itertools.chain.from_iterable(all_rewards))
    # # print(l)
    print(f'length: {len(l)}')
    print('time', time.time() - start)
             

    
    
    # 
    print('done')
    # envs.reset()
    # terminated = False 
    # episode_rewards = np.zeros(n_ens)
    # while not terminated: 
    #     action = envs.action_space.sample()
    #     observation, reward, termination, truncation, info = envs.step(action)
    #     print(termination)
    #     terminated = termination[0]
    
    # print(termination)