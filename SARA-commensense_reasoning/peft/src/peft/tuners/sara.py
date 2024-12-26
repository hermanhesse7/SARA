# coding=utf-8
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
import importlib
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ..utils import PeftConfig, PeftType, transpose


def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


if is_bnb_available():
    import bitsandbytes as bnb



@dataclass
class SARAConfig(PeftConfig):
    """
    Configuration for the SARA PEFT method.
    
    Args:
        r (int): Rank for SARA.
        c (int): Number of components.
        alpha (float): Scaling factor.
        dropout (float): Dropout probability.
        target_modules (Union[List[str], str], optional): Modules to apply SARA to.
        modules_to_save (List[str], optional): Additional modules to save and train.
        merge_weights (bool, default False): Whether to merge SARA weights with the base model in evaluation mode.
    """
    r: int = field(default=8, metadata={"help": "SARA rank"})
    c: int = field(default=2, metadata={"help": "Number of components"})
    alpha: float = field(default=32.0, metadata={"help": "Scaling factor"})
    dropout: float = field(default=0.1, metadata={"help": "Dropout probability"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex pattern to apply SARA. E.g., ['query', 'value'] or '.*SelfAttention.*'"
        },
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Modules to save and keep trainable apart from SARA layers. E.g., ['classifier']."
        },
    )
    merge_weights: bool = field(
        default=False, metadata={"help": "Merge weights of the original model and the SARA model in eval mode"}
    )

    def __post_init__(self):
        self.peft_type = PeftType.SARA


# peft/sara.py

import math
import re
from typing import Optional, Union, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention

from ..utils import transpose, PeftType


class CustomRobertaSelfAttention(RobertaSelfAttention):
    """
    Custom Self-Attention layer for SARA.
    """

    def __init__(self, config):
        super().__init__(config)
        # Initialize any additional components if necessary

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        """
        Forward pass with SARA modifications.
        """
        # Utilize the standard self-attention forward method
        outputs = super().forward(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )

        # Additional SARA-specific computations can be added here if needed

        return outputs


class SHA_DIAGONAL(nn.Module):
    """
    SARA-specific diagonal adapter layer.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        r: int = 8,
        c: int = 2,
        alpha: float = 32.0,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        self.r = r
        self.c = c
        self.alpha = alpha
        self.lora_dropout = nn.Dropout(lora_dropout)

        # Recreate the linear layer and freeze it
        self.pretrained = nn.Linear(in_dim, out_dim, bias=True)
        self.pretrained.weight.requires_grad = False

        # Down projection matrix
        self.down_proj = nn.Linear(in_dim, r, bias=False)

        # Query, Key, Value projection
        self.Wqkv = nn.Linear(r, (r // c) * 3, bias=False)

        # Output projection
        self.Wo = nn.Linear(r // c, r, bias=False)

        # Up projection matrix
        self.up_proj = nn.Linear(r, out_dim, bias=False)

        # Diagonal bias parameter
        self.diagonal_linear_b = nn.Parameter(torch.zeros(out_dim), requires_grad=True)
        nn.init.constant_(self.diagonal_linear_b, 0.001)

        self.scaling = self.alpha / self.r

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = None) -> torch.Tensor:
        """
        Forward pass for the SARA adapter.
        """
        # Pretrained model output
        pretrained_out = self.pretrained(x)

        # Apply dropout
        x_out = self.lora_dropout(x)

        # Down projection
        down_project_out = self.down_proj(x_out)

        B, S, C = down_project_out.shape

        # Compute Q, K, V
        Wqkv = self.Wqkv(down_project_out).reshape(B, S, 3, 1, C // self.c)
        q, k, v = Wqkv.transpose(3, 1).unbind(dim=2)

        # Compute attention scores
        mini_attn_output = torch.matmul(q, k.transpose(-2, -1))
        mini_attn_output = mini_attn_output / math.sqrt(k.size(-1))

        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=mini_attn_output.dtype)
            mini_attn_output = mini_attn_output + attention_mask

        # Apply softmax
        mini_attn_output = F.softmax(mini_attn_output, dim=-1)

        # Apply attention to V
        mini_attn_output = torch.matmul(mini_attn_output, v)

        # Reshape and project
        mini_attn_output = mini_attn_output.transpose(1, 2).reshape(B, S, C // self.c)
        mini_attn_output = self.Wo(mini_attn_output)

        # Up projection
        up_project_out = self.up_proj(mini_attn_output)

        # Apply diagonal bias
        diagonal_b_out = up_project_out * self.diagonal_linear_b
        diagonal_b_out = diagonal_b_out * self.scaling

        # Combine with pretrained output
        return pretrained_out + diagonal_b_out


def mark_only_sara_as_trainable(model: nn.Module, bias: str = "none") -> None:
    """
    Freezes all parameters except those related to SARA.

    Args:
        model (nn.Module): The model to modify.
        bias (str): Bias handling strategy ('none', 'all', 'sara_only').
    """
    for name, p in model.named_parameters():
        if "sara_" not in name:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for name, p in model.named_parameters():
            if "bias" in name:
                p.requires_grad = True
    elif bias == "sara_only":
        for m in model.modules():
            if isinstance(m, SHA_DIAGONAL) and hasattr(m, "diagonal_linear_b") and m.diagonal_linear_b is not None:
                m.diagonal_linear_b.requires_grad = True
    else:
        raise NotImplementedError(f"Bias type '{bias}' is not implemented.")


class SARAModel(torch.nn.Module):
    """
    SARA adapter model that integrates with a pretrained transformers model.
    """

    def __init__(self, config: SARAConfig, model):
        super().__init__()
        self.peft_config = config
        self.model = model
        self._find_and_replace()
        mark_only_sara_as_trainable(self.model, self.peft_config.bias)
        self.forward = self.model.forward

    def _find_and_replace(self):
        """
        Replaces target modules with SARA adapters.
        """
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                if isinstance(target, RobertaSelfAttention):
                    # Replace with CustomRobertaSelfAttention
                    custom_attention = CustomRobertaSelfAttention(target.config)
                    setattr(parent, target_name, custom_attention)
                    # Add SARA adapter to the custom attention layer
                    sara_layer = SHA_DIAGONAL(
                        in_dim=target.output.dense.in_features,
                        out_dim=target.output.dense.out_features,
                        r=self.peft_config.r,
                        c=self.peft_config.c,
                        alpha=self.peft_config.alpha,
                        lora_dropout=self.peft_config.dropout,
                    )
                    setattr(custom_attention, "sara_layer", sara_layer)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _get_submodules(self, key: str) -> Tuple[nn.Module, nn.Module, str]:
        """
        Retrieves the parent module, target module, and target name based on the key.

        Args:
            key (str): The module key.

        Returns:
            Tuple[nn.Module, nn.Module, str]: Parent module, target module, and target name.
        """
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def __getattr__(self, name: str):
        """
        Forwards missing attributes to the wrapped model.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    @property
    def modules_to_save(self):
        """
        Specifies which modules to save.
        """
        return self.peft_config.modules_to_save

    def get_peft_config_as_dict(self, inference: bool = False):
        """
        Returns the PEFT configuration as a dictionary.

        Args:
            inference (bool, optional): Whether to include inference-specific settings.

        Returns:
            dict: Configuration dictionary.
        """
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.peft_config).items()}
        if inference:
            config["inference_mode"] = True
        return config

    def _set_adapter_layers(self, enabled: bool = True):
        """
        Enables or disables SARA adapter layers.

        Args:
            enabled (bool, optional): Whether to enable adapters.
        """
        for module in self.model.modules():
            if isinstance(module, SHA_DIAGONAL):
                module.disable_adapters = not enabled

    def enable_adapter_layers(self):
        """
        Enables SARA adapter layers.
        """
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        """
        Disables SARA adapter layers.
        """
        self._set_adapter_layers(enabled=False)
