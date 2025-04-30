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
    def __init__(self, reg_l1: float, reg_l2: float) -> None:
        super().__init__()
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2

    def forward(self, model: nn.Module):
        elastic_net_loss = 0.0
        for module in model.modules():
            if (
                isinstance(module, nn.Linear)
                or isinstance(module, EinMix)
                or isinstance(module, nn.Conv1d)
                or isinstance(module, nn.Conv2d)
                or isinstance(module, nn.Conv3d)
            ):
                if not module.weight.requires_grad:
                    continue
                elastic_net_loss += (
                    self.reg_l1 * module.weight.flatten().abs().sum()
                    + self.reg_l2 * module.weight.flatten().pow(2).sum()
                )

        return elastic_net_loss
