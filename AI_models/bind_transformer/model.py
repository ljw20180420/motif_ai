from transformers import PretrainedConfig, PreTrainedModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from typing import Union, Tuple
from torchtune.modules import RotaryPositionalEmbeddings, MultiHeadAttention
from .modules import (
    EncoderLayer,
    DecoderLayer,
    ClassificationHead,
)
from .modules.protein_bert import ProteinBERT
from .modules.common import Residual, Duplicate


class BindTransformerConfig(PretrainedConfig):
    """
    model_type: The type of the model.
    label_names: The labels used by trainer as ground truth.
    """

    model_type = "bind_transformer"
    label_names = ["bind"]

    # Config必须有默认值，否则Trainer在保存模型时会报错
    def __init__(
        self,
        protein_vocab: int = None,
        second_vocab: int = None,
        DNA_vocab: int = None,
        max_length: int = None,
        dim_emb: int = None,
        dim_heads: int = None,
        num_heads: int = None,
        depth: int = None,
        chunk_size_feed_forward: int = None,
        dim_ffn: int = None,
        dropout: float = None,
        initializer_range: float = None,
        norm_eps: float = None,
        rotary_value: bool = None,
        pos_weight: float = None,
        seed: int = None,
        **kwargs,
    ) -> None:
        self.protein_vocab = protein_vocab
        self.second_vocab = second_vocab
        self.DNA_vocab = DNA_vocab
        self.max_length = max_length
        self.dim_emb = dim_emb
        self.dim_heads = dim_heads
        self.num_heads = num_heads
        self.depth = depth
        self.chunk_size_feed_forward = chunk_size_feed_forward
        self.dim_ffn = dim_ffn
        self.dropout = dropout
        self.initializer_range = initializer_range
        self.norm_eps = norm_eps
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

        # 二级结构编码器: 蛋白质9种二级结构+锌指结构+KRAB总共11种二级结构的token编码（算上mask token有12个）
        self.second_encode = nn.Sequential(
            nn.Embedding(
                config.second_vocab,
                config.dim_emb,
            ),
            nn.Dropout(config.dropout),
            Residual(
                nn.Sequential(
                    nn.RMSNorm(config.dim_emb, eps=config.norm_eps),
                    Duplicate(
                        MultiHeadAttention(
                            embed_dim=config.dim_emb,
                            num_heads=config.num_heads,
                            num_kv_heads=config.num_heads,
                            head_dim=config.dim_heads,
                            q_proj=nn.Linear(
                                config.dim_emb, config.num_heads * config.dim_heads
                            ),
                            k_proj=nn.Linear(
                                config.dim_emb, config.num_heads * config.dim_heads
                            ),
                            v_proj=nn.Linear(
                                config.dim_emb, config.num_heads * config.dim_heads
                            ),
                            output_proj=nn.Linear(
                                config.num_heads * config.dim_heads, config.dim_emb
                            ),
                            pos_embeddings=RotaryPositionalEmbeddings(
                                dim=config.dim_heads, max_seq_len=config.max_length
                            ),
                            max_seq_len=config.max_length,
                            is_causal=False,
                            attn_dropout=config.dropout,
                        )
                    ),
                )
            ),
            Residual(
                nn.Sequential(
                    nn.RMSNorm(config.dim_emb, eps=config.norm_eps),
                    nn.Linear(config.dim_emb, config.dim_ffn),
                    nn.GELU(),
                    nn.Linear(config.dim_ffn, config.dim_emb),
                    nn.Dropout(config.dropout),
                )
            ),
            nn.RMSNorm(config.dim_emb, eps=config.norm_eps),
        )

        # DNA4个核苷酸的token编码（算上[CLS] token和mask token有6个）
        # 关于[CLS] token，参考BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
        self.DNA_self_attention = nn.Sequential(
            nn.Embedding(
                config.DNA_vocab,
                config.dim_emb,
            ),
            nn.Dropout(config.hidden_dropout),
            Residual(
                nn.Sequential(
                    nn.RMSNorm(config.dim_emb, eps=config.norm_eps),
                    Duplicate(
                        MultiHeadAttention(
                            embed_dim=config.dim_emb,
                            num_heads=config.num_heads,
                            num_kv_heads=config.num_heads,
                            head_dim=config.dim_heads,
                            q_proj=nn.Linear(
                                config.dim_emb, config.num_heads * config.dim_heads
                            ),
                            k_proj=nn.Linear(
                                config.dim_emb, config.num_heads * config.dim_heads
                            ),
                            v_proj=nn.Linear(
                                config.dim_emb, config.num_heads * config.dim_heads
                            ),
                            output_proj=nn.Linear(
                                config.num_heads * config.dim_heads, config.dim_emb
                            ),
                            pos_embeddings=RotaryPositionalEmbeddings(
                                dim=config.dim_heads, max_seq_len=config.max_length
                            ),
                            max_seq_len=config.max_length,
                            is_causal=False,
                            attn_dropout=config.dropout,
                        )
                    ),
                )
            ),
        )

        self.DNA_protein_cross_attention = MultiHeadAttention(
            embed_dim=config.dim_emb,
            num_heads=config.num_heads,
            num_kv_heads=config.num_heads,
            head_dim=config.dim_heads,
            q_proj=nn.Linear(config.dim_emb, config.num_heads * config.dim_heads),
            k_proj=nn.Linear(config.dim_emb, config.num_heads * config.dim_heads),
            v_proj=nn.Linear(config.dim_emb, config.num_heads * config.dim_heads),
            output_proj=nn.Linear(config.num_heads * config.dim_heads, config.dim_emb),
            pos_embeddings=RotaryPositionalEmbeddings(
                dim=config.dim_heads, max_seq_len=config.max_length
            ),
            max_seq_len=config.max_length,
            is_causal=False,
            attn_dropout=config.dropout,
        )

        self.DNA_second_cross_attention = MultiHeadAttention(
            embed_dim=config.dim_emb,
            num_heads=config.num_heads,
            num_kv_heads=config.num_heads,
            head_dim=config.dim_heads,
            q_proj=nn.Linear(config.dim_emb, config.num_heads * config.dim_heads),
            k_proj=nn.Linear(config.dim_emb, config.num_heads * config.dim_heads),
            v_proj=nn.Linear(config.dim_emb, config.num_heads * config.dim_heads),
            output_proj=nn.Linear(config.num_heads * config.dim_heads, config.dim_emb),
            pos_embeddings=RotaryPositionalEmbeddings(
                dim=config.dim_heads, max_seq_len=config.max_length
            ),
            max_seq_len=config.max_length,
            is_causal=False,
            attn_dropout=config.dropout,
        )

        self.DNA_ffn = nn.Sequential(
            Residual(
                nn.Sequential(
                    nn.RMSNorm(config.dim_emb, eps=config.norm_eps),
                    nn.Linear(config.dim_emb, config.dim_ffn),
                    nn.GELU(),
                    nn.Linear(config.dim_ffn, config.dim_emb),
                    nn.Dropout(config.dropout),
                )
            )
        )

        # huggingface的分类头
        self.classifier = nn.Sequential(
            nn.RMSNorm(config.dim_emb, eps=config.norm_eps),
            nn.Linear(config.dim_emb, config.dim_emb),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_emb, 1),
            nn.Flatten(),
        )

        # 交叉熵
        self.loss_fn = BCEWithLogitsLoss(
            reduction="sum", pos_weight=torch.tensor(self.config.pos_weight)
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

        # 先初始化参数, 再读取protein_bert
        # 使用protein_bert编码蛋白
        self.protein_bert = nn.Sequential(
            ProteinBERT(
                num_tokens=26,
                dim=128,
                dim_global=512,
                depth=6,
                narrow_conv_kernel=9,
                wide_conv_kernel=9,
                wide_conv_dilation=5,
                attn_heads=4,
                attn_dim_head=64,
                filename="bind_transformer/modules/epoch_92400_sample_23500000.pkl",
            ),
            nn.Linear(128, config.dim_emb),
            nn.Dropout(config.dropout),
            nn.RMSNorm(config.dim_emb, eps=config.norm_eps),
        )

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
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range, generator=self.generator
            )
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.RMSNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        protein_ids: torch.Tensor,
        second_ids: torch.Tensor,
        DNA_ids: torch.Tensor,
        bind: torch.Tensor = None,
    ):
        """
        蛋白质和DNA的mask token都是0。
        protein_ids: (batch_size, protein_length).
        second_ids: (batch_size, protein_length).
        DNA_ids: (batch_size, DNA_length + 1). +1是因为DNA开头要加[CLS] mask.
        bind: (batch_size, 2). bind[bn, 0]是不结合的概率。bind[bn, 1]是结合的概率。两者相加是1。
        output["logit"]: Binary classification logits (batch_size,).
        output["loss"]: If bind is provided, the loss is cross entropy.
        """
        self.accession_input(protein_ids, second_ids, DNA_ids, bind)

        # 蛋白质氨基酸编码
        protein_embs = self.protein_bert(
            protein_ids,
        )
        # 二级结构编码
        second_embs = self.second_encode(
            second_ids,
        )
        # DNA自注意力编码
        DNA_embs = self.DNA_self_attention(DNA_ids)
        # DNA和氨基酸交叉注意力
        DNA_embs = DNA_embs + self.DNA_protein_cross_attention(
            F.rms_norm(
                DNA_embs, normalized_shape=self.config.dim_emb, eps=self.config.norm_eps
            ),
            protein_embs,
        )
        # DNA和二级结构交叉注意力
        DNA_embs = DNA_embs + self.DNA_second_cross_attention(
            F.rms_norm(
                DNA_embs, normalized_shape=self.config.dim_emb, eps=self.config.norm_eps
            ),
            second_embs,
        )
        # DNA前馈网络
        DNA_embs = self.DNA_ffn(DNA_embs)
        # 分类头
        logits = self.classifier(DNA_embs)

        if bind is not None:
            return {"logit": logits, "loss": self.loss_fn(input=logits, target=bind)}
        return {"logit": logits}

    def accession_input(
        self,
        protein_ids: torch.Tensor,
        second_ids: torch.Tensor,
        DNA_ids: torch.Tensor,
        bind: Union[torch.Tensor, None],
    ) -> None:
        # 检查批处理大小是否一致
        assert (
            protein_ids.shape[0] == second_ids.shape[0]
            and protein_ids.shape[0] == DNA_ids.shape[0]
            and (bind is None or protein_ids.shape[0] == bind.shape[0])
        ), "batch size is not consistent"
        # 检查氨基酸和二级结构长度一致, protein bert会在蛋白序列开头和结尾增加<start>和<end> token, 所以长度增加了2
        assert (
            protein_ids.shape[-1] == second_ids.shape[-1] + 2
        ), "protein sequence length and secondary structure annotation length are not consistent"
        # 检查输入序列长度有没有超标。
        assert protein_ids.shape[-1] <= self.config.max_length, "protein is too long"
        # 检查DNA是否太长
        assert DNA_ids.shape[-1] <= self.config.max_length
