import torch
from torch import nn

class BitRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm
        https://github.com/huggingface/transformers/blob/c5f0288bc7d76f65996586f79f69fba8867a0e67/src/transformers/models/llama/modeling_llama.py#L76C1-L90C59
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super(BitLinear, self).__init__(in_features, out_features, bias, rms_norm_eps=1e-8, bits=8, flg_before_linear=True)
        self.layernorm = BitRMSNorm(hidden_size=in_features, eps=rms_norm_eps)
        self.bits = bits
        self.Qb = 2 ** (self.bits - 1)
        self.flg_before_linear = flg_before_linear

    def absmax_quantize(self, x):
        epsilon = 1e-6
        if self.flg_before_linear:
            gamma = torch.abs(x).max().clamp(min=epsilon)
            x_scaled = x * self.Qb / gamma
            x_q = torch.round(x_scaled).clamp(-self.Qb, self.Qb - 1)
        else:
            eta = x.min()
            gamma = torch.abs(x - eta).max().clamp(min=epsilon)
            x_scaled = (x - eta) * self.Qb / gamma
            x_q = torch.round(x_scaled).clamp(0, self.Qb - 1)
        
        x_q = (x_q - x_scaled).detach() + x_scaled
        return x_q, gamma

    def custom_sign(self, x):
        return (x > 0).to(torch.int8) * 2 - 1

    def quantize_weights(self):
        alpha = self.weight.mean()
        weight_centered = self.weight - alpha
        weight_binarized = self.custom_sign(weight_centered)
        beta = self.weight.abs().mean()
        weight_scaled = weight_centered / (weight_centered.abs().max() + self.epsilon) #weight_centered is divided by weight_centered.abs().max() so that the scale is approximately the same before and after bypass
        weight_binarized = (weight_binarized - weight_scaled).detach() + weight_scaled 
        return weight_binarized, beta
        
    def forward(self, x):
        # layernorm (input: x, output: x_norm)
        x_norm = self.layernorm(x)

        # absmax quatization (input: x_norm, output: x_q,gamma)
        x_q, gamma = self.absmax_quantize(x_norm)

        # 1 bit weights (input: -, output: w_q, beta)
        w_q, beta = self.quantize_weights()

        # tesnor product (input: x_q,gamma, output: x_matmul)
        x_matmul = torch.nn.functional.linear(x_q, w_q, self.bias)

        # dequantization (input: x_matmul,beta,gamma, output: output)
        output = x_matmul * (beta * gamma / self.Qb)

        return output
