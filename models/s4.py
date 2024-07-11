from torch import nn
import math
import torch
from einops import rearrange, repeat

"""
Wrapper for stacking S4D layers (w/ residual connection)
>> Based on https://github.com/state-spaces/s4/blob/main/example.py
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
Minimal version of S4D with extra options and features stripped out, for pedagogical purposes.
>> Based on https://github.com/state-spaces/s4/blob/main/models/s4/s4d.py
"""


class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        # Generate dt
        H = d_model
        log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N//2))
        A_imag = math.pi * repeat(torch.arange(N//2), 'n -> h n', h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = torch.exp(self.log_dt) # (H)
        C = torch.view_as_complex(self.C) # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device) # (H N L)
        C = C * (torch.exp(dtA)-1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

        return K

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4DLayer(nn.Module):
    def __init__(self, d_model, d_state=64, p_dropout=0.0, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p_dropout) if p_dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2*self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u, **kwargs):  # absorbs return_output and transformer src mask
        """ Input and output shape (Bsz, seqLen, HiddenDim) """
        u = u.transpose(-1, -2)  # (B L H) -> (B H L)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L)  # (H L)

        # Convolution
        k_f = torch.fft.rfft(k, n=2 * L)  # (H L)
        u_f = torch.fft.rfft(u, n=2 * L)  # (B H L)
        y = torch.fft.irfft(u_f * k_f, n=2 * L)[..., :L]  # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)

        y = y.transpose(-1, -2)  # -> (B L H)
        return y, None  # We return a dummy state to satisfy this repo's interface, but this can be modified
