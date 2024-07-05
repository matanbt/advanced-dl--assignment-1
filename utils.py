# report number of parameters
from torch import nn


def print_num_params(model: nn.Module):
    """
    Return the number of parameters in the model.
    """
    n_params = sum(p.numel() for p in model.parameters())

    print("[[ ", "number of parameters: %.2fM" % (n_params / 1e6,), " ]]")
    return n_params
