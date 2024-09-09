# import os 
# os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
# os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6

from typing import Callable, List 
import gymnasium as gym
import torch
from ppo_continuous_action import Agent 
import numpy as np
import time 

def evaluate(
    env_fns: List[Callable],
    model_path: str,
    eval_episodes: int,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
):      
    # env_fns = env_fns[:10]
    envs = gym.vector.AsyncVectorEnv(env_fns=env_fns)
    print('environment is created')
    agent = Model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
            next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
            # next_obs, _, _, _, infos = envs.step(actions.numpy())
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    episodic_returns.append(info['cum_reward'])
                    # if "episode" not in info:
                    #     continue
                    # print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                    # episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns

if __name__=="__main__":
    from simulation.market_gym import Market     
    # set up 
    env = 'flow'
    volume = 40
    #
    configs = [{'market_env': env, 'execution_agent': 'rl_agent', 'volume': volume, 'seed': 100+s} for s in range(70)]
    env_fns = [lambda: Market(config) for config in configs]
    # model_path = "runs/Market__ppo_continuous_action__0__1725462757_gaussian_20lots_more_features/ppo_continuous_action.cleanrl_model"
    # model_path = "runs/Market__ppo_continuous_action__0__1725470471_20lots_std3/ppo_continuous_action.cleanrl_model"

    # model_path = 'runs/Market__ppo_continuous_action__0__1725548983_20lots_std3/ppo_continuous_action.cleanrl_model'
    # 40 lots noise 
    # model_path = 'runs/Market__ppo_continuous_action__0__1725552339_20lots_std3/ppo_continuous_action.cleanrl_model'
    # 20 lots flow 
    model_path =  'runs/Market__ppo_continuous_action__0__1725555789_flow_20/ppo_continuous_action.cleanrl_model'
    # 40 lots flow 
    model_path = 'runs/Market__ppo_continuous_action__0__1725559747_flow_40/ppo_continuous_action.cleanrl_model'


    t = time.time()
    returns = evaluate(
        model_path=model_path,
        env_fns=env_fns,
        eval_episodes=7000,
        Model=Agent,
        # device="cpu",
    )
    
    print(np.mean(returns))
    print(f"elapsed time: {time.time()-t}")

    np.savez(f'raw_rewards/rewards_{env}_{volume}_rl_agent.npz', rewards=returns)


