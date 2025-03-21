from transformers import PreTrainedModel, RoFormerConfig, RoFormerModel
import torch.nn as nn
import torch
import torch.nn.functional as F


class BindTransformerConfig(RoFormerConfig):
    """
    model_type: The type of the model.
    label_names: The labels used by trainer as ground truth.
    """

    model_type = "bind_transformer"
    label_names = ["bind"]

    def __init__(
        self,
        vocab_size,
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        intermediate_size,
        hidden_dropout_prob,
        attention_probs_dropout_prob,
        max_position_embeddings,
        pos_weight,
        seed,
        **kwargs,
    ):
        """
        vocab_size: 24 for protein + DNA.
        hidden_size: Model embedding dimension.
        num_hidden_layers: Number of EncoderLayer.
        num_attention_heads: Number of attention heads.
        intermediate_size :FeedForward intermediate dimension size.
        hidden_dropout_prob: The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob: The dropout ratio for the attention probabilities.
        max_position_embeddings: The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 1536).
        pos_weight: Weight for positive samples (https://www.tensorflow.org/tutorials/structured_data/imbalanced_data).
        seed: Random seed for intialization.
        """
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
            **kwargs,
        )


class BindTransformerModel(PreTrainedModel):
    """
    config_class: The corresponding cofiguration class of BindTransformerModel.
    """

    config_class = BindTransformerConfig

    def __init__(self, config):
        """
        config: An instance of the configuration class.
        """
        super().__init__(config)
        self.generator = torch.Generator().manual_seed(config.seed)
        self.model = RoFormerModel(config)
        # The last layer generally does not use nn.LayerNorm or nn.BatchNorm (https://stats.stackexchange.com/questions/361700/lack-of-batch-normalization-before-last-fully-connected-layer).
        # The laste layer generally does not use dropout (https://stackoverflow.com/questions/46841362/where-dropout-should-be-inserted-fully-connected-layer-convolutional-layer).
        # Normalization is applied before activation, while dropout is applied after activation (https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout).
        # CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->

        # Roformer will initialize weights by itsefl, so we only need intialize the output linear layer. For consistency, we use the initializer of Roformer to initializ the output linear layer.
        self.mlp = nn.Linear(in_features=config.hidden_size, out_features=1)
        self.model._initialize_weights(self.mlp)

        # We use cross entropy as loss function, so we do not include output activation layer in the model directly (https://forums.fast.ai/t/lesson-5-mnist-model-missing-softmax-function/54436).

    def forward(self, DNAprotein: torch.Tensor, bind: torch.Tensor = None):
        """
        DNAprotein: The target DNA sequence followed by the protein sequence (batch_size, (DNA_length + protein_length)).
        bind: A value between 0 and 1 to indicate the binding strenth between DNA and protein, 1 for binding and 0 for not binding (batch_size,).
        output["logit"]: Binary classification logits (batch_size,).
        output["loss"]: If bind is provided, the loss is cross entropy.
        """
        batch_size, seq_length = DNAprotein.shape
        assert (
            seq_length <= self.config.max_position_embeddings
        ), f"The length of DNA + protein ({seq_length}) cannot beyond max_position_embeddings ({self.config.max_position_embeddings})."
        # Arbitrarily choose the last position to predict the logits.
        logit = self.mlp(
            self.model(
                input_ids=DNAprotein,
                attention_mask=torch.ones(
                    batch_size,
                    seq_length,
                    dtype=torch.int64,
                    device=self.device,
                ),
            ).last_hidden_state[:, -1, :]
        ).flatten()
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
