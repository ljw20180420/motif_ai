from transformers import PreTrainedModel, RoFormerConfig, RoFormerModel
import torch.nn as nn
import torch
import torch.nn.functional as F


class BindTransformerConfig(RoFormerConfig):
    model_type = "bind_transformer"
    label_names = ["bind"]

    def __init__(
        self,
        vocab_size=24,  # protein + DNA
        hidden_size=256,  # model embedding dimension
        num_hidden_layers=4,  # number of EncoderLayer
        num_attention_heads=4,  # number of attention heads
        intermediate_size=1024,  # FeedForward intermediate dimension size
        hidden_dropout_prob=0.1,  # The dropout probability for all fully connected layers in the embeddings, encoder, and pooler
        attention_probs_dropout_prob=0.1,  # The dropout ratio for the attention probabilities
        max_position_embeddings=64,  # The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 1536).
        pos_weight=1,  # weight for positive samples (https://www.tensorflow.org/tutorials/structured_data/imbalanced_data)
        seed=63036,  # random seed for intialization
        **kwargs
    ):
        self.seed = seed
        self.pos_weight = pos_weight
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            **kwargs
        )


class BindTransformerModel(PreTrainedModel):
    config_class = BindTransformerConfig

    def __init__(self, config):
        super().__init__(config)
        self.generator = torch.Generator().manual_seed(config.seed)
        self.model = RoFormerModel(config)
        self.mlp = nn.Linear(in_features=config.hidden_size, out_features=1)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1, generator=self.generator)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, seq: torch.Tensor, bind: torch.Tensor = None):
        # seq (batch_size X sequence_length)
        # model(seq) (batch_size X sequence_length X hidden_size)
        # model(seq)[:, -1, :] arbitrary choose the last position to predict the logits
        batch_size = seq.shape[0]
        seq_length = seq.shape[1]
        logit = (
            self.mlp(
                self.model(
                    input_ids=seq,
                    attention_mask=torch.ones(
                        batch_size,
                        seq_length,
                        dtype=torch.int64,
                        device=self.model.device,
                    ),
                ).last_hidden_state[:, -1, :]
            )
            .view(batch_size, 1)
            .flatten()
        )
        if bind is not None:
            return {
                "logit": logit,
                "loss": F.binary_cross_entropy_with_logits(
                    input=logit,
                    target=bind,
                    reduction="sum",
                    pos_weight=torch.tensor(self.config.pos_weight),
                ),
            }
        return {"logit": logit}
