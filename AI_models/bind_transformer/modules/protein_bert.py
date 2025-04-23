import torch
import pickle
from torch import nn, einsum

from einops.layers.torch import Rearrange
from einops import rearrange

from .common import Residual


class CrossAttention(nn.Module):
    def __init__(self, dim, dim_global, heads, dim_head):
        super().__init__()
        self.scale = dim_head**-0.5

        self.to_q = nn.Sequential(
            nn.Linear(dim_global, dim_head * heads, bias=False),
            nn.Tanh(),
            Rearrange("b n (h d) -> b h n d", h=heads),
        )
        self.to_k = nn.Sequential(
            nn.Linear(dim, dim_head * heads, bias=False),
            nn.Tanh(),
            Rearrange("b n (h d) -> b h n d", h=heads),
        )
        self.to_v = nn.Sequential(
            nn.Linear(dim, dim_global, bias=False),
            nn.GELU(),
            Rearrange("b n (h d) -> b h n d", h=heads),
        )

    def forward(self, x, context):
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return out


class Layer(nn.Module):
    def __init__(
        self,
        dim,
        dim_global,
        narrow_conv_kernel,
        wide_conv_kernel,
        wide_conv_dilation,
        attn_heads,
        attn_dim_head,
    ):
        super().__init__()

        self.narrow_conv = nn.Sequential(
            Rearrange("b n d -> b d n"),
            nn.Conv1d(
                dim,
                dim,
                narrow_conv_kernel,
                padding=narrow_conv_kernel // 2,
            ),
            Rearrange("b d n -> b n d"),
            nn.GELU(),
        )

        wide_conv_padding = (
            wide_conv_kernel + (wide_conv_kernel - 1) * (wide_conv_dilation - 1)
        ) // 2

        self.wide_conv = nn.Sequential(
            Rearrange("b n d -> b d n"),
            nn.Conv1d(
                dim,
                dim,
                wide_conv_kernel,
                dilation=wide_conv_dilation,
                padding=wide_conv_padding,
            ),
            Rearrange("b n d -> b d n"),
            nn.GELU(),
        )

        self.extract_global_info = nn.Sequential(
            nn.Linear(dim_global, dim),
            nn.GELU(),
        )

        self.local_norm = nn.LayerNorm(dim, eps=1e-3)

        self.local_feedforward = nn.Sequential(
            Residual(
                nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.GELU(),
                )
            ),
            nn.LayerNorm(dim, eps=1e-3),
        )

        self.global_attend_local = CrossAttention(
            dim=dim,
            dim_global=dim_global,
            heads=attn_heads,
            dim_head=attn_dim_head,
        )

        self.global_dense = nn.Sequential(nn.Linear(dim_global, dim_global), nn.GELU())

        self.global_norm = nn.LayerNorm(dim_global, eps=1e-3)

        self.global_feedforward = nn.Sequential(
            Residual(nn.Sequential(nn.Linear(dim_global, dim_global), nn.GELU())),
            nn.LayerNorm(dim_global, eps=1e-3),
        )

    def forward(self, tokens, annotation):
        global_info = self.extract_global_info(annotation)

        # process local (protein sequence)

        narrow_out = self.narrow_conv(tokens)
        wide_out = self.wide_conv(tokens)

        tokens = tokens + narrow_out + wide_out + global_info
        tokens = self.local_norm(tokens)
        tokens = self.local_feedforward(tokens)
        # process global (annotations)

        annotation = (
            annotation
            + self.global_dense(annotation)
            + self.global_attend_local(annotation, tokens)
        )
        annotation = self.global_norm(annotation)
        annotation = self.global_feedforward(annotation)

        return tokens, annotation


# main model


class ProteinBERT(nn.Module):
    def __init__(
        self,
        num_tokens,
        dim,
        dim_global,
        depth,
        narrow_conv_kernel,
        wide_conv_kernel,
        wide_conv_dilation,
        attn_heads,
        attn_dim_head,
        filename,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.global_bias = nn.Parameter()

        self.active_global = nn.Sequential(nn.GELU(), Rearrange("b d -> b () d"))

        self.layers = nn.ModuleList(
            [
                Layer(
                    dim=dim,
                    dim_global=dim_global,
                    narrow_conv_kernel=narrow_conv_kernel,
                    wide_conv_dilation=wide_conv_dilation,
                    wide_conv_kernel=wide_conv_kernel,
                    attn_heads=attn_heads,
                    attn_dim_head=attn_dim_head,
                )
                for layer in range(depth)
            ]
        )

        self.load_weights(filename)

    def forward(self, protein_ids):
        tokens = self.token_emb(protein_ids)

        annotation = self.global_bias[None, :].expand(protein_ids.shape[0], -1)
        annotation = self.active_global(annotation)

        for layer in self.layers:
            tokens, annotation = layer(tokens, annotation)

        # return tokens, rearrange(annotation, "b () d -> b d")
        return tokens

    def load_weights(self, filename):
        with open(filename, "rb") as fd:
            _, model_weights, _ = pickle.load(fd)

        self.global_bias.data = torch.from_numpy(model_weights[1])
        self.token_emb.weight.data = torch.from_numpy(model_weights[2])

        for i, layer in enumerate(self.layers):
            # torch Linear weight is (out_feature, in_feature)
            # tensorflow Linear weight is (in_feature, out_feature)
            layer.extract_global_info[0].weight.data = torch.from_numpy(
                model_weights[i * 23 + 3]
            ).T
            layer.extract_global_info[0].bias.data = torch.from_numpy(
                model_weights[i * 23 + 4]
            )
            # torch Conv weight is (out_channel, in_channel, kernel_dim1, kernel_dim2, ...)
            # tensorflow Conv weight is (kernel_dim1, kernel_dim2, ..., in_channel, out_channel)
            layer.narrow_conv[1].weight.data = torch.from_numpy(
                model_weights[i * 23 + 5]
            ).permute(2, 1, 0)
            layer.narrow_conv[1].bias.data = torch.from_numpy(model_weights[i * 23 + 6])
            layer.wide_conv[1].weight.data = torch.from_numpy(
                model_weights[i * 23 + 7]
            ).permute(2, 1, 0)
            layer.wide_conv[1].bias.data = torch.from_numpy(model_weights[i * 23 + 8])
            layer.local_norm.weight.data = torch.from_numpy(model_weights[i * 23 + 9])
            layer.local_norm.bias.data = torch.from_numpy(model_weights[i * 23 + 10])
            layer.local_feedforward[0].fn[0].weight.data = torch.from_numpy(
                model_weights[i * 23 + 11]
            ).T
            layer.local_feedforward[0].fn[0].bias.data = torch.from_numpy(
                model_weights[i * 23 + 12]
            )
            layer.local_feedforward[1].weight.data = torch.from_numpy(
                model_weights[i * 23 + 13]
            )
            layer.local_feedforward[1].bias.data = torch.from_numpy(
                model_weights[i * 23 + 14]
            )
            layer.global_dense[0].weight.data = torch.from_numpy(
                model_weights[i * 23 + 15]
            ).T
            layer.global_dense[0].bias.data = torch.from_numpy(
                model_weights[i * 23 + 16]
            )
            layer.global_attend_local.to_q[0].weight.data = (
                torch.from_numpy(model_weights[i * 23 + 17])
                .permute(1, 0, 2)
                .reshape(512, -1)
                .T
            )
            layer.global_attend_local.to_k[0].weight.data = (
                torch.from_numpy(model_weights[i * 23 + 18])
                .permute(1, 0, 2)
                .reshape(128, -1)
                .T
            )
            layer.global_attend_local.to_v[0].weight.data = (
                torch.from_numpy(model_weights[i * 23 + 19])
                .permute(1, 0, 2)
                .reshape(128, -1)
                .T
            )
            layer.global_norm.weight.data = torch.from_numpy(model_weights[i * 23 + 20])
            layer.global_norm.bias.data = torch.from_numpy(model_weights[i * 23 + 21])
            layer.global_feedforward[0].fn[0].weight.data = torch.from_numpy(
                model_weights[i * 23 + 22]
            ).T
            layer.global_feedforward[0].fn[0].bias.data = torch.from_numpy(
                model_weights[i * 23 + 23]
            )
            layer.global_feedforward[1].weight.data = torch.from_numpy(
                model_weights[i * 23 + 24]
            )
            layer.global_feedforward[1].bias.data = torch.from_numpy(
                model_weights[i * 23 + 25]
            )
