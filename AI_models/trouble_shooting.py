#!/usr/bin/env python

import os

# 把运行文件夹切换为脚本文件夹
os.chdir(os.path.dirname(os.path.abspath(__file__)))


import torch

from bind_transformer.tensordict_model import (
    BindTransformerConfigTensorDict,
    BindTransformerModelTensorDict,
)
from bind_transformer.model import BindTransformerConfig, BindTransformerModel

config = {
    "protein_vocab": 21,
    "second_vocab": 12,
    "dna_vocab": 6,
    "max_num_tokens": 2700,
    "dim_emb": 128,
    "num_heads": 4,
    "dim_heads": 64,
    "depth": 6,
    "dim_ffn": 256,
    "dropout": 0.05,
    "norm_eps": 1e-5,
    "pos_weight": 1.0,
    "reg_l1": 1e-8,
    "reg_l2": 1e-8,
    "initializer_range": 0.02,
    "seed": 63036,
}

model_tensor_dict = BindTransformerModelTensorDict(
    BindTransformerConfigTensorDict(**config)
)
model = BindTransformerModel(BindTransformerConfig(**config))

second_ids = torch.randint(0, 12, (10,))[None, :]

out_tensor_dict = model_tensor_dict.second_encoder(second_ids=second_ids)
breakpoint()
out = model.second_encoder(second_ids=second_ids)
