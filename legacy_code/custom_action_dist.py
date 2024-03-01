import ray
import ray.rllib.algorithms.ppo as ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import Preprocessor

class MyActionDist(ActionDistribution):
    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return 7  # controls model output feature vector size

    def __init__(self, inputs, model):
        super(MyActionDist, self).__init__(inputs, model)
        assert model.num_outputs == 7

    def sample(self): ...
    def logp(self, actions): ...
    def entropy(self): ...

ModelCatalog.register_custom_action_dist("my_dist", MyActionDist)

ray.init()
algo = ppo.PPO(env="CartPole-v1", config={
    "model": {
        "custom_action_dist": "my_dist",
    },
})