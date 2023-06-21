config = ppo.DEFAULT_CONFIG.copy()
config['gamma'] = 1.0
config['callbacks'] = MyCallbacks

if True:
    # tunable 
    config['sgd_minibatch_size'] = tune.choice([512, 1024]) # default 128
    config['sgd_minibatch_size'] = 1024
    # config['sgd_minibatch_size'] = tune.choice()
    config['lr'] = tune.loguniform(1e-5, 1e-3) # default 5e-5 #0.0005 = 
    # config['lr'] = 0.000225
    # config['num_sgd_iter'] = tune.lograndint(25, 35) # default 30
    # config['num_sgd_iter'] = 10
    config['clip_param'] = tune.uniform(0.25, 0.35) # default 0.3
    # config['clip_param'] = 0.3
    # config['vf_clip_param'] = tune.loguniform(10.0, 30.0) # default 10.0, depends on reward scale
    config['vf_clip_param'] = 30.0
    # config['vf_clip_param'] = 1.55


    # fixed, not default
    config['gamma'] = 1.0
    config['entropy_coeff'] = 0
    config['kl_coeff'] = 0.0
    config['kl_target'] = 1.0 # doesn't matter with kl_coeff == 0
        
    # fixed, default
    config['use_critic'] = True
    config['use_gae'] = True
    config['shuffle_sequences'] = True
    config['lr_schedule'] = None
    config['entropy_coeff_schedule'] = None
    config['grad_clip'] = None
    config['batch_mode'] = "truncate_episodes"
    config['lambda'] = 1.0 # could tune, I think this makes most sense for only terminal rewards but need to check logic

# config['sgd_minibatch_size'] = 256
# config['sgd_minibatch_size'] = 1024
# config['sgd_minibatch_size'] = 512
# fixed sample and batch parameters
# see https://github.com/ray-project/ray/issues/10179
config['rollout_fragment_length'] = 256
config['num_envs_per_worker'] = n_envs_per_worker
# config['train_batch_size'] = 256*num_workers*n_envs_per_worker
config['train_batch_size'] = 256*n_envs_per_worker

# # technical config
# config["num_gpus"] = 1/num_workers
config["num_gpus"] = 1
config["num_workers"] = num_workers
# # config['num_cpus_per_worker'] = 0
# config['framework'] = 'tf2'
# # set to false for debugging
# config['eager_tracing'] = False
config['log_level'] = log_level

# environment  
config['env'] = per.MultiAssetSingleAgentEnv 

# evaluation settings 
config['evaluation_duration'] = 500
config['evaluation_interval'] = 1 


# evaluation config

if self.env_config['simulation']:
    mode = 'sim'
else:
    if self.env_config['train']:
        mode = 'train'
    else:
        raise ValueError('test set used for training')

for penalty in self.constraints:
    self.env_config['inventory_penalty_size'] = penalty
    config['env_config'] = self.env_config 
    # evaluation config 
    eval_env_config = self.env_config.copy()
    eval_env_config['train'] = False
    # eval_env_config['exploration'] = False
    config["evaluation_config"] = {"explore": False, "env_config": eval_env_config}
    # tag, fprob, penalty, train or sim 
    name=f"{self.name}_fprob_{self.env_config['fill_probability']}_penalty_{penalty}_{mode}"
    print(f'tune run name: {name}')
    tune.run("PPO", config=config, stop={"training_iteration":training_iterations}, name=name, local_dir='./tune', checkpoint_at_end=True, # resume="Auto",
    mode="max", checkpoint_freq="5", num_samples=n_samples) 
