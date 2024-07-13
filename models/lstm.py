import torch
from torch import nn


class LSTMEncoder(nn.Module):
    """Implement multi-layer LSTM encoder."""

    def __init__(
            self,
            hidden_dim, num_layers, p_dropout=0.1,
            **kwargs,
    ):
        super().__init__()
        self.lstm_blocks = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for _ in range(num_layers):
            self.lstm_blocks.append(LSTMBlock(hidden_dim))
            self.dropouts.append(nn.Dropout(p_dropout))

    def forward(self, x, **kwargs):
        state = None
        for lstm_block, dropout in zip(self.lstm_blocks, self.dropouts):  # TODO verify correctness
            x, state = lstm_block(x, state)
            x = dropout(x)

        return x


class LSTMBlock(nn.Module):
    """Implement a LSTM block with forget, input, output and cell gates.
    Assumes a simple setting of identical hidden and input dimensions.
    Based on variant shown in
    >> https://machinelearningmastery.com/stacked-long-short-term-memory-networks/
    >> https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
    >> https://en.wikipedia.org/wiki/Long_short-term_memory#LSTM_with_a_forget_gate """

    def __init__(self,
                 hidden_dim,
                 **kwargs):
        super().__init__()

        # Forget Gate
        self.W_f = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U_f = nn.Linear(hidden_dim, hidden_dim)

        # Input/Update Gate
        self.W_i = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U_i = nn.Linear(hidden_dim, hidden_dim)

        # Output Gate
        self.W_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U_o = nn.Linear(hidden_dim, hidden_dim)

        # For Cell Input Activation Vector
        self.W_c = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U_c = nn.Linear(hidden_dim, hidden_dim)

        self.sigma_g = nn.Sigmoid()
        self.sigma_c = self.sigma_h = nn.Tanh()

    def forward(self, x, prev_states=None):
        B, L, H = x.size()

        if prev_states is None:
            h_t, c_t = torch.zeros(B, H), torch.zeros(B, H)
            h_t, c_t = h_t.to(x.device), c_t.to(x.device)
        else:
            h_t, c_t = prev_states

        hidden_states = []  # Store hidden states for each time step

        for t in range(L):
            x_t = x[:, t, :]  # (B, H)

            # Calculate activations based on previous hidden states
            f_t = self.sigma_g(self.W_f(x_t) + self.U_f(h_t))  # forget gate
            i_t = self.sigma_g(self.W_i(x_t) + self.U_i(h_t))  # input gate
            o_t = self.sigma_g(self.W_o(x_t) + self.U_o(h_t))  # output gate
            c_hat_t = self.sigma_c(self.W_c(x_t) + self.U_c(h_t))

            # Calculate new hidden state
            c_t = f_t * c_t + i_t * c_hat_t
            h_t = o_t * self.sigma_h(c_t)

            # Store hidden state
            hidden_states.append(h_t.unsqueeze(1))

        hidden_states = torch.cat(hidden_states, dim=1)  # (B, L, H)

        return hidden_states, (h_t, c_t)


class LSTMTorchEncoder(nn.Module):
    """Implement multi-layer LSTM encoder using PyTorch's nn.LSTM."""

    def __init__(
            self,
            hidden_dim, num_layers, p_dropout=0.1,
            **kwargs,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=p_dropout,
            batch_first=True,
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        return x
