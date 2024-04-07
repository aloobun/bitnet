import torch
from torch import nn

class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super(BitLinear, self).__init__(in_features, out_features, bias)
        
    def forward(self, x):
        # layernorm (input: x, output: x_norm)

        # absmax quatization (input: x_norm, output: x_q,gamma)

        # 1 bit weights (input: -, output: w_q, beta)

        # tesnor product (input: x_q,gamma, output: x_matmul)

        # dequantization (input: x_matmul,beta,gamma, output: output)

        return output
