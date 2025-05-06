import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline


class BindTransformerPipeline(DiffusionPipeline):
    def __init__(self, bind_transformer_model):
        super().__init__()

        self.register_modules(bind_transformer_model=bind_transformer_model)

    @torch.no_grad()
    def __call__(self, batch):
        logits = self.bind_transformer_model(
            batch["protein_ids"].to(self.bind_transformer_model.device),
            batch["second_ids"].to(self.bind_transformer_model.device),
            batch["DNA_ids"].to(self.bind_transformer_model.device),
        )["logit"]
        return F.sigmoid(logits)
