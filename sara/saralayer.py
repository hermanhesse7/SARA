# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose

from .._buffer_dict import BufferDict


class SARALayer(BaseTunerLayer):
    # List all names of layers that may contain adapter weights
    adapter_layer_names = ("sara_attn_lambda", "sara_attn_Wqkv, sara_attn_Wo")
    other_param_names = ("sara_A", "sara_B")

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.alpha = {}
        self.scaling = {}
        self.c = {}
        self.sara_dropout = nn.ModuleDict({})

        self.sara_attn_Wqkv = nn.ModuleDict({})
        self.sara_attn_Wo = nn.ModuleDict({})

        # For storing vector scale
        self.sara_attn_lambda = nn.ParameterDict({})
        

        # Stores a reference to the vera_A/B BufferDict.
        # Set to `None` otherwise to avoid computation with random weights
        self.sara_A: Optional[BufferDict] = None
        self.sara_B: Optional[BufferDict] = None

        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs


    @staticmethod
    def _kaiming_init(tensor: torch.Tensor):
        fan = nn.init._calculate_correct_fan(tensor, mode="fan_in")
        gain = math.sqrt(2.0)
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std
        with torch.no_grad():
            return tensor.uniform_(-bound, bound)
        

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def update_layer(
        self,
        adapter_name,
        sara_A: BufferDict,
        sara_B: BufferDict,
        r,
        sara_dropout,
        init_sara_weights,
        sara_lambda_initial: float = 0.001,
        sara_c: int = 1,
        sara_alpha: int = 1
    ):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r[adapter_name] = r
        if sara_dropout > 0.0:
            sara_dropout_layer = nn.Dropout(p=sara_dropout)
        else:
            sara_dropout_layer = nn.Identity()


        self.c[adapter_name] = sara_c
        self.alpha[adapter_name] = sara_alpha

        self.scaling[adapter_name] = sara_alpha / r

        head_dim = (r // sara_c)

        self.sara_dropout.update(nn.ModuleDict({adapter_name: sara_dropout_layer}))
        # Actual trainable parameters
        self.sara_attn_lambda[adapter_name] = nn.Parameter(torch.ones(self.out_features), requires_grad=True)

        self.sara_attn_Wqkv[adapter_name] = nn.Linear(r, head_dim * 3, bias=False)
        self.sara_attn_Wo[adapter_name] = nn.Linear(head_dim, r, bias=False)
        

        # non trainable references to sara_A/B buffers
        self.vera_A = sara_A
        self.vera_B = sara_B

        if adapter_name not in sara_A:
            # This means that this is not the first SARA adapter. We have to add an entry in the dict for this adapter.
            if len(self.sara_A) < 1:
                raise ValueError(
                    "The `vera_A` and `vera_B` buffers are empty. This should not happen. Please report this issue."
                )
            # we can take any of the existing adapter's parameters, as they should all be identical
            sara_A_param = list(self.sara_A.values())[0]
            sara_B_param = list(self.sara_B.values())[0]

            error_tmpl = (
                "{} has a size of {} but {} or greater is required; this probably happened because an additional VeRA "
                "adapter was added after the first one with incompatible shapes."
            )
            # check input size
            if sara_A_param.shape[1] < self.in_features:
                raise ValueError(error_tmpl.format("sara_A", sara_A_param.shape[1], self.in_features))
            # check output size
            if sara_B_param.shape[0] < self.out_features:
                raise ValueError(error_tmpl.format("sara_B", sara_B_param.shape[0], self.out_features))
            # check r
            error_tmpl = (
                "{} has a size of {} but {} or greater is required; this probably happened because an additional VeRA "
                "adapter with a lower rank was added after the first one; loading the adapters "
                "in reverse order may solve this."
            )
            if sara_A_param.shape[0] < self.r[adapter_name]:
                raise ValueError(error_tmpl.format("sara_A", sara_A_param.shape[0], self.r[adapter_name]))
            if sara_B_param.shape[1] < self.r[adapter_name]:
                raise ValueError(error_tmpl.format("sara_B", sara_B_param.shape[1], self.r[adapter_name]))

            self.vera_A[adapter_name] = sara_A_param
            self.vera_B[adapter_name] = sara_B_param

        if init_sara_weights:
            self.reset_vera_parameters(adapter_name, init_sara_weights, sara_lambda_initial=sara_lambda_initial)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_vera_parameters(self, adapter_name, init_sara_weights, sara_lambda_initial: float = 0.001):
        if init_sara_weights is False:
            return
        
        if adapter_name in self.sara_attn_Wqkv.keys():
            if init_sara_weights is True:
                self._kaiming_init(self.sara_attn_Wqkv[adapter_name].weight)
                self._kaiming_init(self.sara_attn_Wo[adapter_name].weight)
            else:
                raise ValueError(f"Unknown initialization {init_sara_weights=}")       
                 
            with torch.no_grad():
                nn.init.zeros_(self.sara_attn_lambda[adapter_name]).fill_(sara_lambda_initial)         


class Linear(nn.Linear, SARALayer):
    # Vera implemented in a dense layer
    def __init__(
        self,
        base_layer,
        sara_A: BufferDict,
        sara_B: BufferDict,
        adapter_name: str,
        r: int = 0,
        sara_c: int = 4,
        sara_alpha: float = 8.0,        
        sara_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_sara_weights: bool = True,
        sara_lambda_initial: float = 0.001,
        **kwargs,
    ) -> None:
        # this gets the init from nn.Linear's super perspective, i.e. nn.Module.__init__, which should always be called
        super(nn.Linear, self).__init__()
        SARALayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, sara_A, sara_B, r, sara_c, sara_alpha, sara_dropout, init_sara_weights, sara_lambda_initial=sara_lambda_initial)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.sara_lambda.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()

                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.sara_lambda.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        """
        Approximate the delta weight for the given Sara adapter.
        
        Args:
            adapter (str): Name of the adapter.
        
        Returns:
            torch.Tensor: An approximate delta weight tensor.
        """
        sara_A = self.sara_A[adapter]
        sara_B = self.sara_B[adapter]
        sara_attn_Wqkv = self.sara_attn_Wqkv[adapter]
        sara_attn_Wo = self.sara_attn_Wo[adapter]
        sara_attn_lambda = self.sara_attn_lambda[adapter]
        scaling = self.scaling[adapter]
        c = self.c[adapter]

        device = sara_B.weight.device
        dtype = sara_B.weight.dtype

        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        if cast_to_fp32:
            sara_A = sara_A.float()
            sara_B = sara_B.float()
            sara_attn_lambda = sara_attn_lambda.float()
            sara_attn_Wqkv = sara_attn_Wqkv.float()
            sara_attn_Wo = sara_attn_Wo.float()
            scaling = scaling.float()

        # Slice the low-rank matrices to match actual layer size
        sliced_A = sara_A[:, :self.in_features].to(device)
        sliced_B = sara_B[:self.out_features, :].to(device)

        # Compose the approximate delta weight by symbolically folding all components
        # Note: This is NOT an exact delta weight, but a linearized approximation
        with torch.no_grad():
            # Compose: x → A → Wqkv → attn → Wo → B → scaled output
            # Approximate entire chain as a linear mapping
            Wqkv_weight = sara_attn_Wqkv.weight  # [3 * C // c, C]
            Wo_weight = sara_attn_Wo.weight      # [C, 3 * C // c]
            
            # Compose attention block as one linear projection (rough approximation)
            approx_attn_linear = Wo_weight @ Wqkv_weight  # shape: [C, C]
            
            delta_weight_approx = scaling * sara_attn_lambda * (sliced_B @ approx_attn_linear @ sliced_A)

            output_tensor = transpose(delta_weight_approx, self.fan_in_fan_out)

            if cast_to_fp32:
                output_tensor = output_tensor.to(dtype=dtype)

            return output_tensor


    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.sara_lambda.keys():
                    continue

                sara_attn_lambda = self.sara_attn_lambda[active_adapter]
                #lambda_b = self.vera_lambda_b[active_adapter]

                sara_A = self.sara_A[active_adapter]
                sara_B = self.sara_B[active_adapter]

                sara_attn_Wqkv = self.sara_attn_Wqkv[active_adapter]
                sara_attn_Wo = self.sara_attn_Wo[active_adapter]

                # As adapted layers may have different shapes and VeRA contains a single shared pair of A and B matrices,
                # we initialize these matrices with the largest required size for each dimension.
                # During the forward pass, required submatrices are sliced out from the shared vera_A and vera_B.
                sliced_A = sara_A[:, : self.in_features].to(x.device)
                sliced_B = sara_B[: self.out_features, :].to(x.device)

                dropout = self.sara_dropout[active_adapter]
                sara_c = self.c[active_adapter]
                scaling = self.scaling[active_adapter]

                x = F.linear(dropout(x), sliced_A)

                B, S, C = x.shape

                num_heads = 1

                x = sara_attn_Wqkv(x).reshape(B, S, 3, num_heads, C // sara_c)

                q, k, v = x.transpose(3, 1).unbind(dim=2)
        
                attn_output = q @ k.transpose(-2, -1)

                attn_output = attn_output / math.sqrt(k.size(-1))

                attn_output = attn_output.softmax(dim=-1)

                attn_output = attn_output @ v

                attn_output = attn_output.transpose(1, 2).reshape(B, S, C // sara_c)

                attn_output = sara_attn_Wo(attn_output)

                up_project_out = sara_B(attn_output)

                output = up_project_out * sara_attn_lambda

                output = output * scaling                

                result = result + output

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "sara." + rep
