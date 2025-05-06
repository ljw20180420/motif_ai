import torch
import pickle
from torch import nn

# torch does not import opt_einsum as backend by default. import opt_einsum manually will enable it.
from torch.backends import opt_einsum
from einops.layers.torch import Rearrange, EinMix
from einops import rearrange, repeat, einsum

from .common import Residual


class CrossAttention(nn.Module):
    def __init__(self, dim, dim_global, heads, dim_head):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_global = dim_global
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Sequential(
            EinMix(
                "b d_g -> b nhd",
                weight_shape="d_g nhd",
                d_g=dim_global,
                nhd=heads * dim_head,
            ),
            nn.Tanh(),
        )
        self.to_k = nn.Sequential(
            EinMix(
                "b s d -> b s nhd",
                weight_shape="d nhd",
                d=dim,
                nhd=heads * dim_head,
            ),
            nn.Tanh(),
        )
        self.to_v = nn.Sequential(
            EinMix("b s d -> b s d_g", weight_shape="d d_g", d=dim, d_g=dim_global),
            nn.GELU(),
            Rearrange("b s (h d) -> b s h d", h=heads, d=dim_global // heads),
        )

    def forward(self, x, context):
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        sim = (
            einsum(
                rearrange(q, "b (h d_h) -> b h d_h", h=self.heads, d_h=self.dim_head),
                rearrange(
                    k, "b s (h d_h) -> b s h d_h", h=self.heads, d_h=self.dim_head
                ),
                "b h d_h, b s h d_h -> b s h",
            )
            * self.scale
        )
        attn = sim.softmax(dim=1)
        out = einsum(attn, v, "b s h, b s h d -> b h d")
        out = rearrange(out, "b h d -> b (h d)", h=self.heads)
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
            Rearrange("b s d -> b d s", d=dim),
            nn.Conv1d(
                dim,
                dim,
                narrow_conv_kernel,
                padding=narrow_conv_kernel // 2,
            ),
            Rearrange("b d s -> b s d", d=dim),
            nn.GELU(),
        )

        wide_conv_padding = (
            wide_conv_kernel + (wide_conv_kernel - 1) * (wide_conv_dilation - 1)
        ) // 2

        self.wide_conv = nn.Sequential(
            Rearrange("b s d -> b d s", d=dim),
            nn.Conv1d(
                dim,
                dim,
                wide_conv_kernel,
                dilation=wide_conv_dilation,
                padding=wide_conv_padding,
            ),
            Rearrange("b d s -> b s d", d=dim),
            nn.GELU(),
        )

        self.extract_global_info = nn.Sequential(
            EinMix(
                "b d_g -> b d",
                weight_shape="d_g d",
                bias_shape="d",
                d_g=dim_global,
                d=dim,
            ),
            nn.GELU(),
            Rearrange("b d -> b () d", d=dim),
        )

        self.local_feedforward = nn.Sequential(
            nn.LayerNorm(dim, eps=1e-3),
            Residual(
                nn.Sequential(
                    EinMix(
                        "b s d -> b s d0",
                        weight_shape="d d0",
                        bias_shape="d0",
                        d=dim,
                        d0=dim,
                    ),
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

        self.global_dense = nn.Sequential(
            EinMix(
                "b d_g -> b d_g0",
                weight_shape="d_g d_g0",
                bias_shape="d_g0",
                d_g=dim_global,
                d_g0=dim_global,
            ),
            nn.GELU(),
        )

        self.global_feedforward = nn.Sequential(
            nn.LayerNorm(dim_global, eps=1e-3),
            Residual(
                nn.Sequential(
                    EinMix(
                        "b d_g -> b d_g0",
                        weight_shape="d_g d_g0",
                        bias_shape="d_g0",
                        d_g=dim_global,
                        d_g0=dim_global,
                    ),
                    nn.GELU(),
                )
            ),
            nn.LayerNorm(dim_global, eps=1e-3),
        )

    def forward(self, tokens, annotation):
        # process local (protein sequence)
        narrow_out = self.narrow_conv(tokens)
        wide_out = self.wide_conv(tokens)
        global_info = self.extract_global_info(annotation)

        tokens = tokens + narrow_out + wide_out + global_info
        tokens = self.local_feedforward(tokens)
        # process global (annotations)

        annotation = (
            annotation
            + self.global_dense(annotation)
            + self.global_attend_local(annotation, tokens)
        )
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
        self.dim_global = dim_global
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.global_bias = nn.Parameter(torch.empty(512))
        self.active_global = nn.GELU()

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
                for _ in range(depth)
            ]
        )

        self.load_weights(filename)

    def forward(self, protein_ids):
        tokens = self.token_emb(protein_ids)

        annotation = repeat(
            self.global_bias, "d -> b d", b=protein_ids.shape[0], d=self.dim_global
        )
        annotation = self.active_global(annotation)

        for layer in self.layers:
            tokens, annotation = layer(tokens, annotation)

        # return tokens, rearrange(annotation, "b () d -> b d")
        return tokens

    def load_weights(self, filename):
        breakpoint()
        if self.global_bias.is_meta:
            return
        with open(filename, "rb") as fd:
            _, model_weights, _ = pickle.load(fd)
        self.global_bias.data = torch.from_numpy(model_weights[1])
        self.token_emb.weight.data = torch.from_numpy(model_weights[2])

        for i, layer in enumerate(self.layers):
            # torch Linear weight is (out_feature, in_feature)
            # tensorflow Linear weight is (in_feature, out_feature)
            # EinMix weight is (in_feature, out_feature), the same as tensorflow
            # EinMix bias is (1, out_feature)
            layer.extract_global_info[0].weight.data = torch.from_numpy(
                model_weights[i * 23 + 3]
            )
            layer.extract_global_info[0].bias.data = rearrange(
                torch.from_numpy(model_weights[i * 23 + 4]), "of -> 1 of"
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
            layer.local_feedforward[0].weight.data = torch.from_numpy(
                model_weights[i * 23 + 9]
            )
            layer.local_feedforward[0].bias.data = torch.from_numpy(
                model_weights[i * 23 + 10]
            )
            layer.local_feedforward[1].module[0].weight.data = torch.from_numpy(
                model_weights[i * 23 + 11]
            )
            layer.local_feedforward[1].module[0].bias.data = rearrange(
                torch.from_numpy(model_weights[i * 23 + 12], "of -> 1 of")
            )
            layer.local_feedforward[2].weight.data = torch.from_numpy(
                model_weights[i * 23 + 13]
            )
            layer.local_feedforward[2].bias.data = torch.from_numpy(
                model_weights[i * 23 + 14]
            )
            layer.global_dense[0].weight.data = torch.from_numpy(
                model_weights[i * 23 + 15]
            )
            layer.global_dense[0].bias.data = rearrange(
                torch.from_numpy(model_weights[i * 23 + 16], "of -> 1 of")
            )
            layer.global_attend_local.to_q[0].weight.data = (
                torch.from_numpy(model_weights[i * 23 + 17])
                .permute(1, 0, 2)
                .reshape(512, -1)
            )
            layer.global_attend_local.to_k[0].weight.data = (
                torch.from_numpy(model_weights[i * 23 + 18])
                .permute(1, 0, 2)
                .reshape(128, -1)
            )
            layer.global_attend_local.to_v[0].weight.data = (
                torch.from_numpy(model_weights[i * 23 + 19])
                .permute(1, 0, 2)
                .reshape(128, -1)
            )
            layer.global_feedforward[0].weight.data = torch.from_numpy(
                model_weights[i * 23 + 20]
            )
            layer.global_feedforward[0].bias.data = torch.from_numpy(
                model_weights[i * 23 + 21]
            )
            layer.global_feedforward[1].module[0].weight.data = torch.from_numpy(
                model_weights[i * 23 + 22]
            )
            layer.global_feedforward[1].module[0].bias.data = rearrange(
                torch.from_numpy(model_weights[i * 23 + 23], "of -> 1 of")
            )
            layer.global_feedforward[2].weight.data = torch.from_numpy(
                model_weights[i * 23 + 24]
            )
            layer.global_feedforward[2].bias.data = torch.from_numpy(
                model_weights[i * 23 + 25]
            )
