
from all_markets_simulation import Market, config 
import gymnasium as gym 

config['execution_agent'] = 'sl_agent'
print(config)

# Question? how is seeding handled by rllib rollouts ???
# even if all parallel environemnts are seeded the same, i guess it should still be plenty of variation
# probably doesnt matter so much 

# option 1 
def make_env(seed):
    config['seed'] = seed
    # print(config)
    env = Market(config=config)
    return env

def set_seed(seed):
    c = config.copy()
    c['seed'] = seed
    return c


print('#####')

l = [lambda: Market(set_seed(seed)) for seed in [1,2]]
print(l)

l[0]

# envs = gym.vector.AsyncVectorEnv([lambda seed: Market(set_seed(seed)) for seed in [1,2]])

# print(envs)
# print(envs.single_action_space)
# print(envs.single_observation_space)
# print(envs.action_space)
# print(envs.observation_space)

# envs.reset()
# terminated = False 
# while not terminated: 
#     action = envs.action_space.sample()
#     observation, reward, termination, truncation, info = envs.step(action)
#     # print(out)
#     print(termination)
#     terminated = termination[0]

# # print(out)


# # option 2







