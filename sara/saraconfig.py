from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class SARAConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`SaraModel`].

    Paper: 

    Args:
        r (`int`, *optional*, defaults to `256`):
            SARA parameter dimension ("rank"). Choose same values like LoRA ranks here.
        target_modules (`Union[List[str], str]`):
            The names of the modules to apply SARA to. Only linear layers are supported.
        projection_prng_key (`int`):
            Vera PRNG init key. Used for initialising sara_A and sara_B for new models or when loading a checkpoint
            that did not include these projections. Defaults to `0`.
        save_projection (`bool`):
            Whether to save the sara_A / sara_B projections in the state dict alongside per layer sara_lambda and sara_attention
            weights. This will increase the size of the checkpoint, but guarantee that we can reload the checkpoint on
            all system configurations. Defaults to `True`.
        sara_dropout (`float`):
            The dropout probability for SARA layers.
        d_initial (`float`, *optional*, defaults to `0.1`):
            Initial init value for `sara_lambda` vector used when initializing the SARA parameters. Small values
            (<=0.1) are recommended.
        fan_in_fan_out (`bool`):
            Set this to True if the layer to replace stores weight like (fan_in, fan_out). For example, gpt-2 uses
            `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.
        bias (`str`):
            Bias type for SARA. Can be 'none', 'all' or 'sara_only'. If 'all' or 'sara_only', the corresponding biases
            will be updated during training. Be aware that this means that, even when disabling the adapters, the model
            will not produce the same output as the base model would have without adaptation.
        modules_to_save (`List[str]`):
            List of modules apart from SARA layers to be set as trainable and saved in the final checkpoint.
        init_weights (`bool`):
            Whether to initialize the weights of the SARA layers with their default initialization. Don't change this
            setting, except if you know exactly what you're doing.
        layers_to_transform (`Union[List[int],int]`):
            The layer indexes to transform, if this argument is specified, it will apply the SARA transformations on
            the layer indexes that are specified in this list. If a single integer is passed, it will apply the SARA
            transformations on the layer at this index.
        layers_pattern (`Optional[Union[List[str], str]]`):
            The layer pattern name, used only if `layers_to_transform` is different from `None`. This should target the
            `nn.ModuleList` of the model, which is often called `'layers'` or `'h'`.
    """

    r: int = field(default=8, metadata={"help": "SARA attention dimension"})

    sara_alpha: int = field(default=8, metadata={"help": "Determine the Scaling factor"})

    sara_c: int = field(default=4, metadata={"help": "Determine the head dimension in SARA attention head"})

    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with SARA."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'. "
                "Only linear layers are supported."
            )
        },
    )
    projection_prng_key: int = field(
        default=0,
        metadata={
            "help": (
                "SARA PRNG init key. Used for initialising sara_A and sara_B for new models or when loading a "
                "checkpoint that did not include these projections."
            )
        },
    )
    save_projection: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to save the sara_A / sara_B projections in the state dict alongside per layer sara_lambda / "
                "sara_attention weights. This will increase the size of the checkpoint, but guarantee that we can reload "
                "the checkpoint on all system configurations."
            )
        },
    )
    sara_dropout: float = field(default=0.0, metadata={"help": "SARA dropout"})

    sara_lambda_initial: float = field(default=0.001, metadata={"help": "Initial init value for sara_lambda vector."})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for SARA. Can be 'none', 'all' or 'sara_only'"})
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of modules apart from SARA layers to be set as trainable and saved in the final checkpoint. For"
                " example, in Sequence Classification or Token Classification tasks, the final layer"
                " `classifier/score` are randomly initialized and as such need to be trainable and saved."
            )
        },
    )
    init_sara_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the weights of the SARA layers with their default initialization. Don't change "
                "this setting, except if you know exactly what you're doing."
            ),
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": (
                "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers"
                " indexes that are specified inside this list. If a single integer is passed, PEFT will transform only"
                " the layer at this index."
            )
        },
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer "
                "pattern is not in the common layers pattern. This should target the `nn.ModuleList` of the "
                "model, which is often called `'layers'` or `'h'`."
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.SARA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        # check for layers_to_transform and layers_pattern
        if self.layers_pattern and not self.layers_to_transform:
            raise ValueError("When `layers_pattern` is specified, `layers_to_transform` must also be specified. ")
        if not self.save_projection:
            warnings.warn(
                "Specified to not save sara_A and saa_B within the state dictionary, instead they will be restored "
                "using the PRNG key store in `config.projection_prng_key`. Consider setting `config.save_projection` "
                "to `True` to guarantee restoring the checkpoint correctly on all system configurations."
            )
