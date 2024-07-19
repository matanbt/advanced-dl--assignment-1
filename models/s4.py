from torch import nn
import math
import torch
from einops import rearrange, repeat

"""
Wrapper for stacking S4D layers (w/ residual connection)
"""


class S4Encoder(nn.Module):

    def __init__(
            self,
            hidden_dim, num_layers, p_dropout=0.1,
            **kwargs,
    ):
        super().__init__()
        self.s4_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for _ in range(num_layers):
            self.s4_layers.append(S4DLayer(hidden_dim, p_dropout=p_dropout))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(p_dropout))

    def forward(self, x, **kwargs):
        """
        Gets `x` of shape (Bsz, seqLen, HiddenDim)
        Returns `x` of shape (Bsz, seqLen, HiddenDim) and `None`
        """

        for s4_layer, layer_norm, dropout in zip(self.s4_layers, self.layer_norms, self.dropouts):
            z, _ = s4_layer(x)  # Apply S4 layer (returned state is ignored)
            z = dropout(z)  # Apply dropout on the S4's output
            x = layer_norm(z + x)  # Residual connection, then normalize ('post-norm')

        return x


"""
Implementation of S4D layer (and its learnable conv. kernel), for diagonal SSM
https://arxiv.org/pdf/2206.11893, Figure 1 and Listing 1
"""


class S4DKernel(nn.Module):

    def __init__(self, hidden_dim, d_state=64, dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        # Generate dt
        self.register_buffer("log_dt",  # (buffer is not trainable)
                             torch.rand(hidden_dim) * (
                                     math.log(dt_max) - math.log(dt_min)
                             ) + math.log(dt_min))  # Geometrically uniform timescale

        # `A` parameter, S4D-Lin init (real and imaginary parts)
        self.A_imag = nn.Parameter(math.pi * repeat(torch.arange(d_state // 2), 'n -> h n',
                                                    h=hidden_dim))  # [TODO] should be trained with weight decay zero
        self.log_A_real = nn.Parameter(
            torch.log(0.5 * torch.ones(hidden_dim, d_state // 2)))  # [TODO] should be trained with weight decay zero

        # `B` parameter is simply real ones

        # `C` parameter, variance preserving init (real and imaginary parts)
        C = torch.randn(hidden_dim, d_state // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))

    def forward(self, L):
        """Calc the conv. kernel for the S4D layer.
        """
        dt = torch.exp(self.log_dt)  # (H)
        C = torch.view_as_complex(self.C)  # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H N)

        # Vandermonde multiplication
        dA = A * dt.unsqueeze(-1)  # (H N)
        K = dA.unsqueeze(-1) * torch.arange(L, device=A.device)  # (H N L)
        C = C * (torch.exp(dA) - 1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

        return K


class S4DLayer(nn.Module):
    def __init__(self, hidden_dim, d_state=64, p_dropout=0.0, **kernel_args):
        super().__init__()

        self.h = hidden_dim
        self.n = d_state

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, d_state=self.n, **kernel_args)

        # Post-convolution layers
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p_dropout) if p_dropout > 0.0 else nn.Identity()
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2 * self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u, **kwargs):
        """ Input and output shape (Bsz, seqLen, HiddenDim) """
        u = u.transpose(-1, -2)  # (B L H) -> (B H L)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L)  # (H L)

        # Convolve y = u * K using FFT
        k_f = torch.fft.rfft(k, n=2 * L)  # (H L)
        u_f = torch.fft.rfft(u, n=2 * L)  # (B H L)
        y = torch.fft.irfft(u_f * k_f, n=2 * L)[..., :L]  # (B H L)

        # Compute D term in state space equation (essentially a skip connection)
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)

        y = y.transpose(-1, -2)  # -> (B L H)
        return y, None  # We return a dummy state to satisfy our repo's interface, but this can be modified
