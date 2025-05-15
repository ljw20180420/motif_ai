import torch
import subprocess
import pandas as pd
from io import StringIO
import os
import numpy as np
from typing import Union
from datasets import Dataset
from .tokenizers import DNA_Tokenizer, Protein_Bert_Tokenizer, Second_Tokenizer


class DataCollator:
    def __init__(
        self,
        proteins: list[str],
        seconds: list[str],
        zinc_nums: list[int],
        minimal_unbind_summit_distance: int,
        select_worst_neg_loss_ratio: float,
        neg_loss: Union[np.ndarray, None],
        dna_length: int,
        max_num_tokens: int,
        seed: int,
    ) -> None:
        assert (
            neg_loss is not None
            or neg_loss is None
            and select_worst_neg_loss_ratio == 0.0
        ), "if neg_loss is None, then select_worst_neg_loss_ratio must be 0.0"
        self.proteins = proteins
        self.seconds = seconds
        self.zinc_nums = zinc_nums
        self.minimal_unbind_summit_distance = minimal_unbind_summit_distance
        self.select_worst_neg_loss_ratio = select_worst_neg_loss_ratio
        self.neg_loss = neg_loss
        self.dna_tokenizer = DNA_Tokenizer(dna_length)
        self.protein_tokenizer = Protein_Bert_Tokenizer(max_num_tokens)
        self.second_tokenizer = Second_Tokenizer()
        self.rng = np.random.default_rng(seed)

    def select_neg_idx(self, unbind_distance: np.ndarray, rns: np.ndarray):
        unbind_mask = np.logical_or(
            unbind_distance == -1,
            unbind_distance >= self.minimal_unbind_summit_distance,
        )
        unbind_rows, unbind_cols = unbind_mask.nonzero()
        if self.neg_loss is None:
            idx = np.arange(len(unbind_rows), dtype=np.int64)
        else:
            idx = self.neg_loss[rns][unbind_mask].argsort()
        worest_num = int(len(rns) * self.select_worst_neg_loss_ratio)
        random_num = len(rns) - worest_num

        neg_idx = np.concat(
            [
                idx[-worest_num:] if worest_num > 0 else np.array([], dtype=np.int64),
                self.rng.choice(
                    idx[:-worest_num] if worest_num > 0 else idx,
                    random_num,
                    replace=False,
                ),
            ]
        )

        return unbind_rows, unbind_cols, neg_idx

    def train_eval_test(self, examples: list[dict]) -> dict[torch.Tensor]:
        unbind_distance = np.array([example["distance"] for example in examples])
        rns = np.array([example["rn"] for example in examples])
        unbind_rows, unbind_cols, neg_idx = self.select_neg_idx(unbind_distance, rns)

        results = {}
        results["dna_ids"] = self.dna_tokenizer(
            [example["dna"] for example in examples]
            + [examples[row]["dna"] for row in unbind_rows[neg_idx]],
            [self.zinc_nums[example["index"]] for example in examples]
            + [self.zinc_nums[col] for col in unbind_cols[neg_idx]],
        )
        results["protein_ids"] = self.protein_tokenizer(
            [self.proteins[example["index"]] for example in examples]
            + [self.proteins[col] for col in unbind_cols[neg_idx]]
        )
        results["second_ids"] = self.second_tokenizer(
            [self.seconds[example["index"]] for example in examples]
            + [self.seconds[col] for col in unbind_cols[neg_idx]],
        )
        results["bind"] = torch.tensor(
            [1.0] * len(examples) + [0.0] * len(examples), dtype=torch.float32
        )
        results["rows"] = rns[unbind_rows[neg_idx]]
        results["cols"] = unbind_cols[neg_idx]

        return results

    def inference(self, examples: list[dict]) -> dict[torch.Tensor]:
        results = {}
        results["dna_ids"] = self.dna_tokenizer(
            [example["dna"] for example in examples],
            [self.zinc_nums[example["index"]] for example in examples],
        )
        results["protein_ids"] = self.protein_tokenizer(
            [self.proteins[example["index"]] for example in examples]
        )
        results["second_ids"] = self.second_tokenizer(
            [self.seconds[example["index"]] for example in examples]
        )

        return results

    def __call__(self, examples: list[dict]) -> dict[torch.Tensor]:
        if "distance" not in examples[0].keys():
            return self.inference(examples)
        return self.train_eval_test(examples)

    def neg_map(self, batch):
        unbind_rows, unbind_cols, neg_idx = self.select_neg_idx(
            np.array(batch["distance"]), np.array(batch["rn"])
        )

        return {
            "index": batch["index"] + unbind_cols[neg_idx].tolist(),
            "dna": batch["dna"] + [batch["dna"][row] for row in unbind_rows[neg_idx]],
            "bind": [1.0] * len(batch["index"]) + [0.0] * len(neg_idx),
        }


@torch.no_grad()
def train_validation_test_split(
    ds, validation_ratio, test_ratio, seed, unique_test=False
):
    ds = ds["train"].train_test_split(
        test_size=test_ratio + validation_ratio, seed=seed
    )
    ds2 = ds["test"].train_test_split(
        test_size=test_ratio / (test_ratio + validation_ratio),
        seed=seed,
    )
    ds["validation"] = ds2["train"]
    ds["test"] = ds2["test"]
    if unique_test:
        # remove test examples which appear in train and valid examples
        with open("bind_transformer/temp_train_eval.txt", "w") as fd:
            fd.write("\n".join(ds["train"]["dna"] + ds["validation"]["dna"]))
        with open("bind_transformer/temp_test.txt", "w") as fd:
            fd.write("\n".join(ds["test"]["dna"]))
        output = subprocess.run(
            [
                "grep",
                "-vnf",
                "bind_transformer/temp_train_eval.txt",
                "bind_transformer/temp_test.txt",
            ],
            capture_output=True,
        )
        os.remove("bind_transformer/temp_train_eval.txt")
        os.remove("bind_transformer/temp_test.txt")
        df = pd.read_csv(
            StringIO(output.stdout.decode()), sep=":", names=["index", "dna"]
        )
        ds["test"] = ds["test"].select(df["index"] - 1)
    return ds
