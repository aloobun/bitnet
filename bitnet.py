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
        super(BitLinear, self).__init__(in_features, out_features, bias, rms_norm_eps=1e-8)
        self.layernorm = BitRMSNorm(hidden_size=in_features, eps=rms_norm_eps)
        
    def forward(self, x):
        # layernorm (input: x, output: x_norm)
        x_norm = self.layernorm(x)

        # absmax quatization (input: x_norm, output: x_q,gamma)

        # 1 bit weights (input: -, output: w_q, beta)

        # tesnor product (input: x_q,gamma, output: x_matmul)

        # dequantization (input: x_matmul,beta,gamma, output: output)

        return output
