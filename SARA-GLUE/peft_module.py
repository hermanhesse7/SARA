import math
from typing import List, Tuple, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import config
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention

class CustomRobertaSelfAttention(RobertaSelfAttention):
    def __init__(self, config):
        super().__init__(config)

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
        is_cross_attention = encoder_hidden_states is not None

        # Decide which attention mask to use
        attention_mask_for_query = encoder_attention_mask if is_cross_attention else attention_mask

        # Pass attention_mask to self.query and self.value
        mixed_query_layer = self.query(hidden_states, attention_mask=attention_mask_for_query)

        if is_cross_attention:
            # Cross-attention
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states, attention_mask=encoder_attention_mask))
            attention_mask = encoder_attention_mask
        else:
            # Self-attention
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states, attention_mask=attention_mask))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # The rest of your attention computations remain the same
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Continue with the rest of the forward method
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs



class SHA_DIAGONAL(nn.Module):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        r: int = config.CONFIG.r,
        c: int = config.CONFIG.c,
        alpha: int = config.CONFIG.alpha,
        lora_dropout: float = config.CONFIG.dropout,
    ):
        super().__init__()
        self.r = r
        self.num_heads = 1
        self.c = c
        self.head_dim = (self.r // self.c) // self.num_heads
        self.alpha = alpha
        self.lora_dropout = nn.Dropout(lora_dropout)

        # recreate the linear layer and freeze it (the actual weight values will be copied in outside of this class)
        self.pretrained = nn.Linear(in_dim, out_dim, bias=True)
        self.pretrained.weight.requires_grad = False

        # create the down projection matrix and initialize with same method as in Hugging Face PEFT library
        self.down_proj = nn.Linear(in_dim, r, bias=False)
        #self._kaiming_init(self.down_proj.weight, generator=torch.manual_seed(config.CONFIG.seed))
        
        self.Wqkv = nn.Linear(r, (r // self.c)*3, bias=False)
        #nn.init.constant_(self.Wqkv.weight , 0)
        
        self.Wo = nn.Linear(r // self.c, r, bias=False)
        #nn.init.constant_(self.Wo.weight , 0)
        
        # create the up projection matrix and initialize to zero
        self.up_proj = nn.Linear(r, out_dim, bias=False)
        #self._kaiming_init(self.up_proj.weight, generator=torch.manual_seed(config.CONFIG.seed))

        # Add the custom DiagonalLinear layer
        self.diagonal_linear_b = nn.Parameter(torch.zeros(out_dim), requires_grad=True)
        nn.init.constant_(self.diagonal_linear_b, 0.001)

        self.scaling = self.alpha / self.r
    

    def _kaiming_init(self, tensor: torch.Tensor, generator: torch.Generator):

        fan = nn.init._calculate_correct_fan(tensor, mode="fan_in")
        gain = math.sqrt(2.0)
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std
        with torch.no_grad():
            return tensor.uniform_(-bound, bound, generator=generator)


    def forward(self, x, attention_mask=None):    
        pretrained_out = self.pretrained(x)

        x_out = self.lora_dropout(x)

        down_project_out = self.down_proj(x_out)

        B, S, C = down_project_out.shape

        Wqkv = self.Wqkv(down_project_out).reshape(B, S, 3, self.num_heads, C // self.c)

        q, k, v = Wqkv.transpose(3, 1).unbind(dim=2)
        
        mini_attn_output = q @ k.transpose(-2, -1)

        mini_attn_output = mini_attn_output / math.sqrt(k.size(-1))

        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=mini_attn_output.dtype)
            mini_attn_output = mini_attn_output + attention_mask

        mini_attn_output = mini_attn_output.softmax(dim=-1)

        mini_attn_output = mini_attn_output @ v

        mini_attn_output = mini_attn_output.transpose(1, 2).reshape(B, S, C // self.c)

        mini_attn_output = self.Wo(mini_attn_output)

        up_project_out = self.up_proj(mini_attn_output)

        diagonal_b_out = up_project_out * self.diagonal_linear_b

        diagonal_b_out = diagonal_b_out * self.scaling

        return pretrained_out + diagonal_b_out



def freeze_model(model):
    for name, param in model.named_parameters():
        if "Wqkv" not in name and "Wo" not in name and "diagonal_linear_b" not in name and "classifier" not in name:
            param.requires_grad = False


def create_peft(module):
    """Converts a linear module to a peft linear module."""
    k, d = module.weight.shape  # pytorch nn.Linear weights are transposed, that is why shape is (k, d) and not (d, k)
    peft = SHA_DIAGONAL(in_dim=d, out_dim=k)
    with torch.no_grad():
        peft.pretrained.weight.copy_(module.weight)
        peft.pretrained.bias.copy_(module.bias)
    return peft   



def add_peft_layers(
    model,
    module_names: Tuple=("query", "value"),
    ignore_layers: List[int]=[]
):
    module_types: Tuple=(nn.Linear,)

    # disable dropout in frozen layers
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    # replace chosen linear modules with lora modules
    for name, module in model.named_children():
        if isinstance(module, module_types) and name in module_names:
            temp_peft = create_peft(module)
            setattr(model, name, temp_peft)
        else:
            ignore_layers_str = [str(i) for i in ignore_layers]
            if name not in ignore_layers_str:
                add_peft_layers(module, module_names, ignore_layers)      