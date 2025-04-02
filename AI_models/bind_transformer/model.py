from transformers import PretrainedConfig, PreTrainedModel
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from typing import Union
import math
from transformers.models.roformer.modeling_roformer import (
    RoFormerSinusoidalPositionalEmbedding,
)
from .modules import (
    EmbeddingLayerNormDropOut,
    EncoderLayer,
    DecoderLayer,
    ClassificationHead,
)


class BindTransformerConfig(PretrainedConfig):
    """
    model_type: The type of the model.
    label_names: The labels used by trainer as ground truth.
    """

    model_type = "bind_transformer"
    label_names = ["bind"]

    def __init__(
        self,
        protein_animo_acids_vocab_size: int,
        protein_secondary_structure_vocab_size: int,
        protein_coarse_grained_size: int,
        protein_max_position_embeddings: int,
        DNA_vocab_size: int,
        DNA_max_position_embeddings: int,
        embedding_size: int,
        hidden_size: int,
        num_attention_heads: int,
        num_hidden_layers: int,
        chunk_size_feed_forward: int,
        intermediate_size: int,
        hidden_act: str,
        hidden_dropout_prob: float,
        attention_probs_dropout_prob: float,
        initializer_range: float,
        layer_norm_eps: float,
        rotary_value: bool,
        pos_weight: float,
        seed: int,
        **kwargs,
    ):
        self.protein_animo_acids_vocab_size = protein_animo_acids_vocab_size
        self.protein_secondary_structure_vocab_size = (
            protein_secondary_structure_vocab_size
        )
        self.protein_coarse_grained_size = protein_coarse_grained_size
        self.protein_max_position_embeddings = protein_max_position_embeddings
        self.DNA_vocab_size = DNA_vocab_size
        self.DNA_max_position_embeddings = DNA_max_position_embeddings
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.chunk_size_feed_forward = chunk_size_feed_forward
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.rotary_value = rotary_value
        self.pos_weight = pos_weight
        self.seed = seed
        super().__init__(
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
        # 蛋白质20个氨基酸的token编码（算上mask token有21个）
        self.protein_animo_acids_embedding = EmbeddingLayerNormDropOut(
            config.protein_animo_acids_vocab_size,
            config.embedding_size,
            config.layer_norm_eps,
            config.hidden_dropout_prob,
        )
        # 蛋白质9种二级结构+锌指结构+KRAB总共11种二级结构的token编码（算上mask token有12个）
        self.protein_secondary_structure_embedding = EmbeddingLayerNormDropOut(
            config.protein_secondary_structure_vocab_size,
            config.embedding_size,
            config.layer_norm_eps,
            config.hidden_dropout_prob,
        )

        # 把蛋白质嵌入向量投影到蛋白质隐层向量
        self.protein_project = nn.Linear(config.embedding_size, config.hidden_size)
        # 蛋白质位置编码
        self.protein_sinusoidal = RoFormerSinusoidalPositionalEmbedding(
            math.ceil(
                config.protein_max_position_embeddings
                / config.protein_coarse_grained_size
            ),
            config.hidden_size // config.num_attention_heads,
        )

        # 蛋白质Encoder
        self.encoder = nn.ModuleList(
            [
                EncoderLayer(
                    config.num_attention_heads,
                    config.hidden_size,
                    config.attention_probs_dropout_prob,
                    config.rotary_value,
                    config.intermediate_size,
                    config.hidden_act,
                    config.layer_norm_eps,
                    config.hidden_dropout_prob,
                    config.chunk_size_feed_forward,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

        # DNA4个核苷酸的token编码（算上[CLS] token和mask token有6个）
        # 关于[CLS] token，参考BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
        self.DNA_embedding = EmbeddingLayerNormDropOut(
            config.DNA_vocab_size,
            config.embedding_size,
            config.layer_norm_eps,
            config.hidden_dropout_prob,
        )

        # 把DNA嵌入向量投影到DNA隐层向量
        self.DNA_project = nn.Linear(config.embedding_size, config.hidden_size)
        # DNA位置编码
        self.DNA_sinusoidal = RoFormerSinusoidalPositionalEmbedding(
            config.DNA_max_position_embeddings,
            config.hidden_size // config.num_attention_heads,
        )

        # DNA Decoder
        self.decoder = nn.ModuleList(
            [
                DecoderLayer(
                    config.num_attention_heads,
                    config.hidden_size,
                    config.attention_probs_dropout_prob,
                    config.rotary_value,
                    config.intermediate_size,
                    config.hidden_act,
                    config.layer_norm_eps,
                    config.hidden_dropout_prob,
                    config.chunk_size_feed_forward,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

        # huggingface的分类头
        self.classifier = ClassificationHead(
            self.config.hidden_size,
            self.config.hidden_dropout_prob,
            self.config.num_labels,
            self.config.hidden_act,
        )
        # 交叉熵
        self.loss_fn = CrossEntropyLoss(
            weight=torch.tensor([1, self.config.pos_weight]), reduction="sum"
        )
        # 设置随机生成子，让训练可重复
        self.generator = torch.Generator().manual_seed(config.seed)
        # post_init
        #     init_weights
        #         prune_heads
        #         _initialize_weights
        #             _init_weights
        #         tie_weights
        #     _backward_compatibility_gradient_checkpointing
        self.post_init()

    def _init_weights(self, module):
        # 从roformer抄的初始化方法。增加了self.generator来固定随机生成。让参数初始化可重复。
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range, generator=self.generator
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, RoFormerSinusoidalPositionalEmbedding):
            pass
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range, generator=self.generator
            )
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        protein_animo_acids_ids: torch.Tensor,
        protein_secondary_structure_ids: torch.Tensor,
        DNA_ids: torch.Tensor,
        bind: torch.Tensor = None,
    ):
        """
        蛋白质和DNA的mask token都是0。
        protein_animo_acids_ids: (batch_size, protein_length).
        protein_secondary_structure_ids: (batch_size, protein_length).
        DNA_ids: (batch_size, DNA_length + 1). +1是因为DNA开头要加[CLS] mask.
        bind: (batch_size, 2). bind[bn, 0]是不结合的概率。bind[bn, 1]是结合的概率。两者相加是1。
        output["logit"]: Binary classification logits (batch_size,).
        output["loss"]: If bind is provided, the loss is cross entropy.
        """
        self.accession_input(
            protein_animo_acids_ids, protein_secondary_structure_ids, DNA_ids, bind
        )

        # 蛋白质氨基酸编码+二级结构编码。
        protein_embeddings = self.protein_animo_acids_embedding(
            protein_animo_acids_ids,
        ) + self.protein_secondary_structure_embedding(
            protein_secondary_structure_ids,
        )
        # 叠加连续protein_coarse_grained_size个位置的蛋白质编码
        protein_embeddings = protein_embeddings.view(
            protein_embeddings.shape[0],
            -1,
            self.config.protein_coarse_grained_size,
            protein_embeddings.shape[-1],
        ).sum(dim=2)
        # DNA碱基编码
        DNA_embeddings = self.DNA_embedding(DNA_ids)
        # 把蛋白质嵌入向量投影到蛋白质隐层向量
        protein_embeddings = self.protein_project(protein_embeddings)
        # 把DNA嵌入向量投影到DNA隐层向量
        DNA_embeddings = self.DNA_project(DNA_embeddings)
        # 蛋白质位置编码 [batch_size, protein_coarse_grained_length] -> [batch_size, num_heads, protein_coarse_grained_length, embedding_size_per_head]
        protein_sinusoidal_pos = self.protein_sinusoidal(protein_embeddings[:-1])[
            None, None, :, :
        ]
        # DNA位置编码 [batch_size, DNA_length] -> [batch_size, num_heads, DNA_length, embedding_size_per_head]
        DNA_sinusoidal_pos = self.DNA_sinusoidal(DNA_embeddings[:-1])[None, None, :, :]
        # 根据mask token（0）计算attention mask
        protein_mask = self.get_coarse_grained_protein_mask(protein_animo_acids_ids)
        protein_self_attention_mask = self.get_zeros_attention_mask(
            protein_mask, protein_mask
        )
        DNA_mask = self.get_DNA_mask(DNA_ids)
        DNA_self_attention_mask = self.get_zeros_attention_mask(DNA_mask, DNA_mask)
        protein_DNA_cross_attention_mask = self.get_zeros_attention_mask(
            protein_mask, DNA_mask
        )
        # 计算蛋白质encoder
        for layer in self.encoder:
            protein_embeddings = layer(
                protein_embeddings, protein_self_attention_mask, protein_sinusoidal_pos
            )
        # 计算DNA decoder
        for layer in self.decoder:
            DNA_embeddings = layer(
                DNA_embeddings,
                protein_embeddings,
                DNA_self_attention_mask,
                protein_DNA_cross_attention_mask,
                DNA_sinusoidal_pos,
            )
        # 计算对数得分
        logits = self.classifier(DNA_embeddings)
        if bind is not None:
            return {"logit": logits, "loss": self.loss_fn(input=logits, target=bind)}
        return {"logit": logits}

    def accession_input(
        self,
        protein_animo_acids_ids: torch.Tensor,
        protein_secondary_structure_ids: torch.Tensor,
        DNA_ids: torch.Tensor,
        bind: Union[torch.Tensor, None],
    ) -> None:
        # 检查批处理大小是否一致
        assert (
            protein_animo_acids_ids.shape[0] == protein_secondary_structure_ids.shape[0]
            and protein_animo_acids_ids.shape[0] == DNA_ids.shape[0]
            and (bind is None or protein_animo_acids_ids.shape[0] == bind.shape[0])
        ), "batch size is not consistent"
        # 检查氨基酸和二级结构长度一致
        assert (
            protein_animo_acids_ids.shape[-1]
            == protein_secondary_structure_ids.shape[-1]
        ), "protein sequence length and secondary structure annotation length are not consistent"
        # 用mask_id（0）补齐输入蛋白序列到coarse grained的整数倍，并且检查输入序列长度有没有超标。
        protein_animo_acids_ids = self.padding_zeros_to_coarse_grained(
            protein_animo_acids_ids
        )
        protein_secondary_structure_ids = self.padding_zeros_to_coarse_grained(
            protein_secondary_structure_ids
        )
        assert (
            protein_animo_acids_ids.shape[-1]
            <= self.config.protein_max_position_embeddings
        ), "protein is too long"
        # 检查DNA是否太长
        assert DNA_ids.shape[-1] <= self.config.DNA_max_position_embeddings

    def padding_zeros_to_coarse_grained(
        self, protein_ids: torch.Tensor
    ) -> torch.Tensor:
        padding_size = protein_ids.shape[-1] % self.config.protein_coarse_grained_size
        if padding_size > 0:
            return torch.cat(
                [
                    protein_ids,
                    torch.zeros(
                        protein_ids.shape[0],
                        padding_size,
                        dtype=protein_ids.dtype,
                    ),
                ],
                dim=1,
            )
        return protein_ids

    def get_coarse_grained_protein_mask(self, protein_ids):
        return protein_ids.view(protein_ids.shape[0], -1, 5).any(dim=2).to(self.dtype)

    def get_DNA_mask(self, DNA_ids):
        return DNA_ids.to(torch.bool).to(self.dtype)

    @staticmethod
    def get_zeros_attention_mask(
        query_mask: torch.Tensor, key_mask: torch.Tensor
    ) -> torch.Tensor:
        return (
            query_mask[:, :, None]
            * key_mask[:, None, :]
            * torch.finfo(query_mask.dtype).min
        )
