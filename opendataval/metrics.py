import torch
import torch.nn.functional as F

from opendataval.util import FuncEnum


def accuracy(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute accuracy of two one-hot encoding tensors."""
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    a = a.to(device)
    b = b.to(device)
    return (a.argmax(dim=1) == b.argmax(dim=1)).float().mean().item()


def neg_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return -torch.square(a - b).sum().sqrt().item()


def neg_mse(a: torch.Tensor, b: torch.Tensor):
    return -F.mse_loss(a, b).item()


class Metrics(FuncEnum):
    ACCURACY = FuncEnum.wrap(accuracy)
    NEG_L2 = FuncEnum.wrap(neg_l2)
    NEG_MSE = FuncEnum.wrap(neg_mse)
