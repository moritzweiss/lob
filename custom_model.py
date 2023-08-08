import logging
import numpy as np
import gymnasium as gym

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
from ray.rllib.utils.deprecation import deprecation_warning
from ray.util import log_once
from ray.rllib.models.action_dist import ActionDistribution
from typing import Optional
from ray.rllib.utils.typing import TensorType, List, Union, Tuple, ModelConfigDict
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from torch.nn import Softplus

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)


class CustomTorchModel(TorchModelV2, nn.Module):
    """Generic fully connected network."""

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **customized_model_kwargs
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # hiddens = list(model_config.get("fcnet_hiddens", [])) + list(
        #     model_config.get("post_fcnet_hiddens", [])
        # )
        # activation = model_config.get("fcnet_activation")
        # if not model_config.get("fcnet_hiddens", []):
        #     activation = model_config.get("post_fcnet_activation")
        # no_final_linear = model_config.get("no_final_linear")
        # self.vf_share_layers = model_config.get("vf_share_layers")
        # self.free_log_std = model_config.get("free_log_std")
        # # Generate free-floating bias variables for the second half of
        # # the outputs.
        # if self.free_log_std:
        #     assert num_outputs % 2 == 0, (
        #         "num_outputs must be divisible by two",
        #         num_outputs,
        #     )
        #     num_outputs = num_outputs // 2

        # layers = []
        # prev_layer_size = int(np.product(obs_space.shape))
        # self._logits = None

        # # Create layers 0 to second-last.
        # for size in hiddens[:-1]:
        #     layers.append(
        #         SlimFC(
        #             in_size=prev_layer_size,
        #             out_size=size,
        #             initializer=normc_initializer(1.0),
        #             activation_fn=activation,
        #         )
        #     )
        #     prev_layer_size = size
        
        
        # # The last layer is adjusted to be of size num_outputs, but it's a
        # # layer with activation.
        # if no_final_linear and num_outputs:
        #     layers.append(
        #         SlimFC(
        #             in_size=prev_layer_size,
        #             out_size=num_outputs,
        #             initializer=normc_initializer(1.0),
        #             activation_fn=activation,
        #         )
        #     )
        #     prev_layer_size = num_outputs
        # # Finish the layers with the provided sizes (`hiddens`), plus -
        # # iff num_outputs > 0 - a last linear layer of size num_outputs.
        # else:
        #     if len(hiddens) > 0:
        #         layers.append(
        #             SlimFC(
        #                 in_size=prev_layer_size,
        #                 out_size=hiddens[-1],
        #                 initializer=normc_initializer(1.0),
        #                 activation_fn=activation,
        #             )
        #         )
        #         prev_layer_size = hiddens[-1]
        #     if num_outputs:
        #         self._logits = SlimFC(
        #             in_size=prev_layer_size,
        #             out_size=num_outputs,
        #             initializer=normc_initializer(0.01),
        #             activation_fn=None,
        #         )
        #     else:
        #         self.num_outputs = ([int(np.product(obs_space.shape))] + hiddens[-1:])[
        #             -1
        #         ]

        # # Layer to add the log std vars to the state-dependent means.
        # if self.free_log_std and self._logits:
        #     self._append_free_log_std = AppendBiasLayer(num_outputs)

        # layers = []

        # first layer 
        
        initializer=normc_initializer(1.0)
        linear = nn.Linear(obs_space.shape[0], 128, bias=True)
        initializer(linear.weight)
        activation = nn.Tanh()
        first_layer = nn.Sequential(linear, activation)

        # second layer 
        linear = nn.Linear(128, 128, bias=True)
        initializer(linear.weight)
        activation = nn.Tanh()
        second_layer = nn.Sequential(linear, activation)

        # hidden layers 
        self._hidden_layers = nn.Sequential(first_layer, second_layer)

        # output layer
        initializer=normc_initializer(0.01)
        linear = nn.Linear(128, num_outputs, bias=True)
        initializer(linear.weight)
        activation = nn.Tanh()
        self._logits = nn.Sequential(linear, activation)


        # value layer 
        initializer=normc_initializer(0.01)
        linear = nn.Linear(128, num_outputs, bias=True)
        initializer(linear.weight)
        self._value_branch = linear 

        # Holds the current "base" output (before logits layer).
        self._features = None
        # Holds the last input, in case value branch is separate.
        self._last_flat_in = None
        self._value_branch_separate = None 

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        obs = input_dict["obs_flat"].float()
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._features = self._hidden_layers(self._last_flat_in)
        logits = self._logits(self._features) 
        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        if self._value_branch_separate:
            out = self._value_branch(
                self._value_branch_separate(self._last_flat_in)
            ).squeeze(1)
        else:
            out = self._value_branch(self._features).squeeze(1)
        return out
    

class TorchDirichlet(TorchDistributionWrapper):
    """Dirichlet distribution for continuous actions that are between
    [0,1] and sum to 1.

    e.g. actions that represent resource allocation."""

    def __init__(self, inputs, model):
        """Input is a tensor of logits. The exponential of logits is used to
        parametrize the Dirichlet distribution as all parameters need to be
        positive. An arbitrary small epsilon is added to the concentration
        parameters to be zero due to numerical error.

        See issue #4440 for more details.
        """
        self.epsilon = torch.tensor(1e-7).to(inputs.device)
        layer = Softplus()
        concentration = layer(inputs) + 1.0
        self.dist = torch.distributions.dirichlet.Dirichlet(
            concentration=concentration,
            validate_args=True,
        )
        super().__init__(concentration, model)

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        self.last_sample = nn.functional.softmax(self.dist.concentration, dim=-1)
        return self.last_sample

    @override(ActionDistribution)
    def logp(self, x):
        # Support of Dirichlet are positive real numbers. x is already
        # an array of positive numbers, but we clip to avoid zeros due to
        # numerical errors.
        x = torch.max(x, self.epsilon)
        x = x / torch.sum(x, dim=-1, keepdim=True)
        return self.dist.log_prob(x)

    @override(ActionDistribution)
    def entropy(self):
        return self.dist.entropy()

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space, model_config):
        return np.prod(action_space.shape, dtype=np.int32)