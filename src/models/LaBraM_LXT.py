from lxt.efficient.patches import (
    patch_method,
    replace_module, 
    identity_rule_implicit, 
    layer_norm_forward,
    dropout_forward,
    non_linear_forward,
)
from lxt.efficient.rules import stop_gradient, divide_gradient
from lxt.efficient import monkey_patch

from zennit.composites import LayerMapComposite, SpecialFirstLayerMapComposite
import zennit.rules as z_rules


from models.modeling_finetune import Attention, Mlp, Block, NeuralTransformer, DropPath
from torch.nn import Dropout, LayerNorm
from functools import partial 
from torch.nn import GroupNorm, GELU, Conv2d, Conv1d
import torch
import torch.nn.functional as F

def mlp_forward(self, x):
        # Hopefully unneccesary due to patch to GELU
        x = self.fc1(x)
        x = identity_rule_implicit(self.act, x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x

def group_norm_forward(self, x):
    """
    On normalization operations, we apply the identity rule.
    It is implemented here by stopping the gradient flow through the variance calculation,
    which is equivalent to the identity rule in a Gradient*Input framework.
    """
    input_shape = x.shape
    assert self.num_channels == input_shape[1], f"Expected {self.num_channels} channels, but got {input_shape[1]}"
    assert self.num_channels % self.num_groups == 0, f"Number of channels {self.num_channels} is not divisible by number of groups {self.num_groups}"
    # group separation
    x = x.view(x.shape[0], self.num_groups, self.num_channels//self.num_groups, *input_shape[2:])
    # groupwise normalization statistics
    # Account for all spatial dimensions
    dims = (2,) + tuple(range(3, x.dim()))
    mean = x.mean(dim=dims, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=dims, keepdim=True)
    std = (var + self.eps).sqrt()
    # realization of LRP-Identity rule: normalize while stopping gradient through std
    y = (x - mean) / stop_gradient(std)

    # reshape back to original shape
    y = y.view(input_shape) # Just to be sure that reshapes are properly reversed

    # apply channel-wise transformation
    affine_shape = [1, self.num_channels] + [1] * (y.dim() - 2) # Set all additional dimensions to 1
    # There may be a more efficient broadcasting-oriented way to do this.
    if self.weight is not None:
        weight = self.weight.view(*affine_shape)
        y *= weight
    if self.bias is not None:
        bias = self.bias.view(*affine_shape)
        y += bias
    return y
    
#TODO: Attention
def forward_LaBraM_Attention(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) (B, H, N, C)

        if self.q_norm is not None:
            q = self.q_norm(q).type_as(v)
        if self.k_norm is not None:
            k = self.k_norm(k).type_as(v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = divide_gradient(attn, 2) ### <--- LRP

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if return_attention:
            return attn
            
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = divide_gradient(x, 2) ### <--- LRP

        x = self.proj(x)
        x = self.proj_drop(x)

        if return_qkv:
            return x, qkv

        return x

attnLRP = {
    # Mlp: partial(patch_method, mlp_forward),
    Dropout: partial(patch_method, dropout_forward),
    DropPath: partial(patch_method, dropout_forward),
    LayerNorm: partial(patch_method, layer_norm_forward),
    GroupNorm: partial(patch_method, group_norm_forward),
    GELU: partial(patch_method, non_linear_forward, keep_original=True),
    Attention: partial(patch_method, forward_LaBraM_Attention),
}

zennit_composite = SpecialFirstLayerMapComposite([
    (Conv1d, z_rules.Gamma()),
    (Conv2d, z_rules.Gamma()),
],
first_map=[
     (Conv1d, z_rules.WSquare())
]
)