from torch import nn
from torch import linalg as LA
from einops.layers.torch import EinMix


class Residual(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x) + x


class Elastic_Net(nn.Module):
    def __init__(self, reg_L1: float, reg_L2: float) -> None:
        super().__init__()
        self.reg_L1 = reg_L1
        self.reg_L2 = reg_L2

    def forward(self, model: nn.Module):
        elastic_net_loss = 0.0
        for module in model.children():
            if (
                isinstance(module, nn.Linear)
                or isinstance(module, EinMix)
                or isinstance(module, nn.Conv1d)
                or isinstance(module, nn.Conv2d)
                or isinstance(module, nn.Conv3d)
            ):
                if module.weight.grad is None:
                    continue
                elastic_net_loss += self.reg_L1 * LA.norm(
                    module.weight.flatten(), ord=1
                ) + self.reg_L2 * LA.norm(module.weight.flatten(), ord=2)

        return elastic_net_loss
