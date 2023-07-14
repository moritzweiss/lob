import ray
from ray import air, tune

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print


# algo = (
#     PPOConfig()
#     .rollouts(num_rollout_workers=1)
#     .resources(num_gpus=0)
#     .environment(env="CartPole-v1")
#     .build()
# )

# ray.init()

config = PPOConfig().training(lr=tune.grid_search([0.01, 0.001, 0.0001]))
config.rollouts(num_rollout_workers=1)
config.resources(num_gpus=0)
config.environment(env="CartPole-v1")

tuner = tune.Tuner(
    "PPO",
    run_config=air.RunConfig(
        stop={"episode_reward_mean": 150},
    ),
    param_space=config.to_dict(),
)

tuner.fit()