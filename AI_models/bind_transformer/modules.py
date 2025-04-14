from transformers import activations, pytorch_utils
import torch.nn as nn
import torch
import math
from typing import Union, Tuple


class EmbeddingLayerNormDropOut(nn.Module):
    """Construct the embeddings from word and token_type embeddings."""

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        layer_norm_eps: float,
        hidden_dropout_prob: float,
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(
            vocab_size,
            embedding_size,
        )

        self.LayerNorm = nn.LayerNorm(embedding_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # (batch_size, sequence_length) -> (batch_size, sequence_length, embedding_size)
        return self.dropout(self.LayerNorm(self.embeddings(input_ids)))


class AddAndNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        layer_norm_eps: float,
        hidden_dropout_prob: float,
    ) -> None:
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        # 注意resnet结构
        return self.LayerNorm(self.dropout(self.dense(hidden_states)) + input_tensor)


class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.act_fn = activations.ACT2FN[hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.act_fn(self.dense(hidden_states))


class Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        attention_probs = nn.functional.softmax(
            torch.matmul(query, key.transpose(-1, -2))
            / math.sqrt(self.attention_head_size)
            + attention_mask,
            dim=-1,
        )

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, value)

        return self.reverse_transpose_for_scores(context)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, sequence_length, hidden_size) -> (batch_size, num_attention_heads, sequence_length, attention_head_size)
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def reverse_transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, num_attention_heads, sequence_length, attention_head_size) -> (batch_size, sequence_length, hidden_size)
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (self.hidden_size,)
        return x.view(*new_x_shape)

    @staticmethod
    def apply_rotary_position_embeddings(
        sinusoidal_pos: torch.Tensor,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: Union[torch.Tensor, None] = None,
    ) -> Tuple[torch.Tensor, ...]:
        # https://kexue.fm/archives/8265
        # sin [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        # cos [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        sin, cos = sinusoidal_pos.chunk(2, dim=-1)
        # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
        # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
        # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
        rotate_half_query_layer = torch.stack(
            [-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1
        ).reshape_as(query_layer)
        query_layer = query_layer * cos_pos + rotate_half_query_layer * sin_pos
        # rotate_half_key_layer [-k1,k0,-k3,k2......,-kd-1,kd-2]
        rotate_half_key_layer = torch.stack(
            [-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1
        ).reshape_as(key_layer)
        key_layer = key_layer * cos_pos + rotate_half_key_layer * sin_pos
        if value_layer is not None:
            # rotate_half_value_layer [-v1,v0,-v3,v2......,-vd-1,vd-2]
            rotate_half_value_layer = torch.stack(
                [-value_layer[..., 1::2], value_layer[..., ::2]], dim=-1
            ).reshape_as(value_layer)
            value_layer = value_layer * cos_pos + rotate_half_value_layer * sin_pos
            return query_layer, key_layer, value_layer
        return query_layer, key_layer


class SelfAttention(Attention):
    def __init__(
        self,
        num_attention_heads: int,
        hidden_size: int,
        attention_probs_dropout_prob: float,
        rotary_value: bool,
    ) -> None:
        super().__init__(hidden_size, num_attention_heads, attention_probs_dropout_prob)

        self.rotary_value = rotary_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        sinusoidal_pos: torch.Tensor,
    ) -> torch.Tensor:
        query = self.transpose_for_scores(self.query(hidden_states))
        key = self.transpose_for_scores(self.key(hidden_states))
        value = self.transpose_for_scores(self.value(hidden_states))
        if self.rotary_value:
            query, key, value = self.apply_rotary_position_embeddings(
                sinusoidal_pos, query, key, value
            )
        else:
            query, key = self.apply_rotary_position_embeddings(
                sinusoidal_pos, query, key
            )
        return self.attention(query, key, value, attention_mask)


class CrossAttention(Attention):
    def __init__(
        self,
        num_attention_heads: int,
        hidden_size: int,
        attention_probs_dropout_prob: float,
    ) -> None:
        super().__init__(hidden_size, num_attention_heads, attention_probs_dropout_prob)

    def forward(
        self,
        hidden_states_query: torch.Tensor,
        hidden_states_key_value: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        query = self.transpose_for_scores(self.query(hidden_states_query))
        key = self.transpose_for_scores(self.key(hidden_states_key_value))
        value = self.transpose_for_scores(self.value(hidden_states_key_value))

        return self.attention(query, key, value, attention_mask)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        hidden_size: int,
        attention_probs_dropout_prob: float,
        rotary_value: bool,
        intermediate_size: int,
        hidden_act: str,
        layer_norm_eps: float,
        hidden_dropout_prob: float,
        chunk_size_feed_forward: int,
    ) -> None:
        super().__init__()
        self.self_attention = SelfAttention(
            num_attention_heads, hidden_size, attention_probs_dropout_prob, rotary_value
        )
        self.self_attention_add_and_norm = AddAndNorm(
            hidden_size, hidden_size, layer_norm_eps, hidden_dropout_prob
        )
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_act)
        self.feed_forward_add_and_norm = AddAndNorm(
            hidden_size, intermediate_size, layer_norm_eps, hidden_dropout_prob
        )
        self.chunk_size_feed_forward = chunk_size_feed_forward
        self.seq_len_dim = 1

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        sinusoidal_pos: torch.Tensor,
    ) -> torch.Tensor:
        output = self.self_attention(hidden_states, attention_mask, sinusoidal_pos)
        hidden_states = self.self_attention_add_and_norm(output, hidden_states)

        return pytorch_utils.apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            hidden_states,
        )

    def feed_forward_chunk(self, hidden_states):
        output = self.feed_forward(hidden_states)
        return self.feed_forward_add_and_norm(output, hidden_states)


class DecoderLayer(EncoderLayer):
    def __init__(
        self,
        num_attention_heads: int,
        hidden_size: int,
        attention_probs_dropout_prob: float,
        rotary_value: bool,
        intermediate_size: int,
        hidden_act: str,
        layer_norm_eps: float,
        hidden_dropout_prob: float,
        chunk_size_feed_forward: int,
    ) -> None:
        super().__init__(
            num_attention_heads,
            hidden_size,
            attention_probs_dropout_prob,
            rotary_value,
            intermediate_size,
            hidden_act,
            layer_norm_eps,
            hidden_dropout_prob,
            chunk_size_feed_forward,
        )
        self.cross_attention = CrossAttention(
            num_attention_heads, hidden_size, attention_probs_dropout_prob
        )
        self.cross_attention_add_and_norm = AddAndNorm(
            hidden_size, hidden_size, layer_norm_eps, hidden_dropout_prob
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_key_value: torch.Tensor,
        self_attention_mask: torch.Tensor,
        cross_attention_mask: torch.Tensor,
        sinusoidal_pos: torch.Tensor,
    ) -> torch.Tensor:
        output = self.self_attention(hidden_states, self_attention_mask, sinusoidal_pos)
        hidden_states = self.self_attention_add_and_norm(output, hidden_states)
        output = self.cross_attention(
            hidden_states, hidden_states_key_value, cross_attention_mask
        )
        hidden_states = self.cross_attention_add_and_norm(output, hidden_states)

        return pytorch_utils.apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            hidden_states,
        )


class ClassificationHead(nn.Module):
    # 从huggingface抄的分类头
    def __init__(
        self,
        hidden_size: int,
        hidden_dropout_prob: float,
        num_labels: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_labels)
        self.act_fn = activations.ACT2FN[hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 选择[CLS] token
        x = hidden_states[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
