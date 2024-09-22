# Copyright (c) 2024, EleutherAI
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
import torch
import torch.nn as nn
from torch.nn import LayerNorm as LayerNorm
from .fused_layer_norm import MixedFusedLayerNorm



def get_norm(neox_args):
    if neox_args.norm == "rmsnorm":
        norm = RMSNorm
        eps = neox_args.rms_norm_epsilon
    elif neox_args.norm == "layernorm":
        eps = neox_args.layernorm_epsilon
        norm = MixedFusedLayerNorm if neox_args.layernorm_fusion else LayerNorm
    elif neox_args.norm == "scalenorm":
        eps = neox_args.scalenorm_epsilon
        norm = ScaleNorm
    elif neox_args.norm == "crmsnorm":
        norm = CRMSNorm
        eps = neox_args.crms_norm_epsilon
    # elif neox_args.norm == "jslayernorm":
    #     norm = JSLayerNorm
    #     eps = neox_args.jslayer_norm_epsilon
    else:
        raise ValueError(f"norm {neox_args.norm} not recognized")
    return norm, eps


decorator = torch.jit.script

@decorator
def crms_norm(x, eps: float):
    discarded_element = x.sum(dim=-1, keepdim=True)
    return x * torch.rsqrt((x.square().sum(dim=-1, keepdim=True) + discarded_element.square()) / (x.shape[-1] + 1) + eps)


class CRMSNorm(torch.nn.Module):
    def __init__(self, dim, p=-1.0, eps=1e-8, bias=False):
        """
            Compressed Root Mean Square Layer Normalization
        :param dim: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        """
        super(CRMSNorm, self).__init__()

        self.eps = eps
        self.d = dim
        self.p = p
        self.bias = bias

        self.scale = torch.nn.Parameter(torch.ones(dim))
        self.register_parameter("scale", self.scale)

    def forward(self, x):
        return self.scale * crms_norm(x.float(),self.eps).type_as(x)

class RMSNorm(torch.nn.Module):
    def __init__(self, dim, p=-1.0, eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param dim: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = dim
        self.p = p
        self.bias = bias

        self.scale = torch.nn.Parameter(torch.ones(dim))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = torch.nn.Parameter(torch.zeros(dim))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        dtype = x.dtype
        if self.p < 0.0 or self.p > 1.0:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return (self.scale * x_normed + self.offset).to(dtype)
        # print("RMS Norm Shapes")
        # print(f"scale param:{self.scale.shape}, normed tensor:{x_normed.shape}")
        return (self.scale * x_normed).to(dtype)
        # return (x_normed).to(dtype)
   

# class RMSNorm(torch.nn.Module):
#     def __init__(self, dim, p=-1.0, eps=1e-8, bias=False):
#         super(RMSNorm, self).__init__()
#         self.eps = eps
#         self.scale = torch.nn.Parameter(torch.Tensor(1, dim).fill_(1.0))
#         self.bias = None
#         if bias:
#             self.bias = torch.nn.Parameter(torch.Tensor(1, dim).fill_(0.0))

#     def forward(self, x):
#         dtype = x.dtype
#         #root_n = x.shape[-1]**0.5
#         x = x.float()
#         norm = torch.norm(x, p=2, dim=-1, keepdim=True) #/ root_n
#         norm = norm / torch.sqrt(torch.tensor(x.size(-1), dtype=torch.float32).to(x.device))
#         if self.bias is not None:
#             return (self.bias + (x / (norm + self.eps) * self.scale)).to(dtype)
#         return (x / (norm + self.eps) * self.scale).to(dtype)
    

class ScaleNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.g = torch.nn.Parameter(torch.ones(1))
        self.eps = eps

    def forward(self, x):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x / n * self.g





# # @torch.jit.script
# def js_layer_norm(x: torch.Tensor, eps: float, c: int):
#     x_mean = x.mean(dim=1)
#     x_var = x.var(dim=1, correction=0)

#     x_mean_var = x_mean.var(dim=-1, correction=0)
#     x_var_var = x_var.var(dim=-1, correction=0)

#     mean_nm = (c - 2) * x_mean_var
#     mean_dm = torch.norm(x_mean, p=2, dim=-1, keepdim=True) + eps
#     mean_shrinkage = 1 - (mean_nm / mean_dm)
#     mean_shrinkage = torch.clamp(mean_shrinkage, min=0, max=1)

#     x_js_mean = mean_shrinkage * x_mean

#     var_nm = (c - 2) * x_var_var
#     var_dm = torch.norm(x_var, p=2, dim=-1, keepdim=True) + eps
#     var_shrinkage = 1 - (var_nm / var_dm)
#     var_shrinkage = torch.clamp(var_shrinkage, min=0, max=1)

#     x_js_var = var_shrinkage * x_var

#     if (torch.any(torch.isnan(x_mean_var)) or torch.any(torch.isnan(x_var_var)) or
#     torch.any(torch.isnan(mean_nm)) or torch.any(torch.isnan(mean_dm)) or
#     torch.any(torch.isnan(var_nm)) or torch.any(torch.isnan(var_dm)) or
#     torch.any(torch.isnan(mean_shrinkage)) or torch.any(torch.isnan(var_shrinkage))):
    
#         print("NaN detected in variables. Debugging information:")
#         print(f"x_mean_var: {x_mean_var}","\n",f" x_var_var: {x_var_var}")
#         print(f"mean_nm: {mean_nm}","\n",f" mean_dm: {mean_dm}")
#         print(f"var_nm: {var_nm}","\n",f" var_dm: {var_dm}")
#         print(f"mean_shrinkage: {mean_shrinkage}","\n",f" var_shrinkage: {var_shrinkage}")
#         exit()

#     return (x - x_js_mean) * torch.rsqrt(x_js_var + eps)

# def js_layer_norm(x: torch.Tensor, eps: float, c: int):
#     x_mean = x.mean(dim=-1)
#     x_var = x.var(dim=-1, correction=0)

#     num_features = x.shape[-1]
#     shrinkage_factor = 1 - (num_features - 2) / (torch.sum(x_var) + eps)
#     shrinkage_factor = torch.clamp(shrinkage_factor, min=0)

#     shrunk_mean = (x_mean * shrinkage_factor).unsqueeze(-1)
#     shrunk_var = (x_var * shrinkage_factor).unsqueeze(-1)

#     if (torch.any(torch.isnan(x_mean)) or torch.any(torch.isnan(x_var)) or
#     torch.any(torch.isnan(shrinkage_factor)) or torch.any(torch.isnan(shrunk_mean)) or
#     torch.any(torch.isnan(shrunk_var))):
    
#         print("NaN detected in variables. Debugging information:")
#         print(f"x_mean: {x_mean}","\n",f" x_var: {x_var}")
#         print(f"shrinkage_factor: {shrinkage_factor}","\n",f" shrunk_mean: {shrunk_mean}")
#         print(f"shrunk_var: {shrunk_var}")
#         exit()

#     return (x - shrunk_mean) / torch.sqrt(shrunk_var + eps)

# def js_layer_norm(x: torch.Tensor, eps: float, c: int):
#     x_mean = x.mean(dim=-1, keepdim=True)
#     x_var = x.var(dim=-1, keepdim=True, unbiased=False)

#     x_mean_var = x_mean.var(dim=1, keepdim=True, unbiased=False)
#     x_var_var = x_var.var(dim=1, keepdim=True, unbiased=False)

#     mean_nm = (c - 2) * x_mean_var
#     mean_dm = (torch.norm(x_mean, p=2, dim=1, keepdim=True) + eps)**2
#     mean_shrinkage = 1 - (mean_nm / mean_dm)


#     x_js_mean = mean_shrinkage * x_mean

#     var_nm = (c - 2) * x_var_var
#     var_dm = (torch.norm(x_var, p=2, dim=1, keepdim=True) + eps)**2
#     var_shrinkage = 1 - (var_nm / var_dm)


#     x_js_var = var_shrinkage * x_var

#     return_val =  (x - x_js_mean) * torch.rsqrt(x_js_var + eps)

#     print("x is : ", x)
#     print(f"x_mean_var: {x_mean_var}","\n",f" x_var_var: {x_var_var}")
#     print(f"mean_nm: {mean_nm}","\n",f" mean_dm: {mean_dm}")
#     print(f"var_nm: {var_nm}","\n",f" var_dm: {var_dm}")
#     print(f"mean_shrinkage: {mean_shrinkage}","\n",f" var_shrinkage: {var_shrinkage}")
#     print("x_js_var is : ", x_js_var)
#     print("return_val is : ", return_val)
    
#     if (torch.any(torch.isnan(x_mean_var)) or torch.any(torch.isnan(x_var_var)) or
#     torch.any(torch.isnan(mean_nm)) or torch.any(torch.isnan(mean_dm)) or
#     torch.any(torch.isnan(var_nm)) or torch.any(torch.isnan(var_dm)) or
#     torch.any(torch.isnan(mean_shrinkage)) or torch.any(torch.isnan(var_shrinkage)) or 
#     torch.any(torch.isnan(mean_shrinkage)) or torch.any(torch.isnan(var_shrinkage))):
#         print("############################################################################################################# \n\n\n")
#         print("NaN detected in variables. Debugging information:")
#         print("x is : ", x)
#         print(f"x_mean_var: {x_mean_var}","\n",f" x_var_var: {x_var_var}")
#         print(f"mean_nm: {mean_nm}","\n",f" mean_dm: {mean_dm}")
#         print(f"var_nm: {var_nm}","\n",f" var_dm: {var_dm}")
#         print(f"mean_shrinkage: {mean_shrinkage}","\n",f" var_shrinkage: {var_shrinkage}")
#         print("x_js_var is : ", x_js_var)
#         print("return_val is : ", return_val)
#         exit()


#     return return_val

# class JSLayerNorm(nn.Module):
#     def __init__(self, hidden_dim: int, eps=1e-8) -> None:
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones((1, 1, hidden_dim)))
#         self.bias = nn.Parameter(torch.zeros((1, 1, hidden_dim)))
#         self.eps = eps

#     def forward(self, x):
#         x_js_standardized = js_layer_norm(x, self.eps, x.shape[-1])
#         return (x_js_standardized * self.weight) + self.bias