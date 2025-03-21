import torch
from diffusers import DiffusionPipeline


class BindTransformerPipeline(DiffusionPipeline):
    def __init__(self, bind_transformer_model):
        super().__init__()

        self.register_modules(bind_transformer_model=bind_transformer_model)

    @torch.no_grad()
    def __call__(self, batch):
        return self.bind_transformer_model(
            batch["DNAprotein"].to(self.bind_transformer_model.device)
        )["logit"]
