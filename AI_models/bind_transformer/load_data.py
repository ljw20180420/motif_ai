import torch
import numpy as np
from .model import BindTransformerConfig

outputs_train = ["DNA", "protein"] + BindTransformerConfig.label_names
outputs_test = ["DNA", "protein"] + BindTransformerConfig.label_names
outputs_inference = ["DNA", "protein"]


# seq="ACGT"
def DNA_tokenizer(seq):
    return (
        torch.from_numpy(np.frombuffer(seq.encode(), dtype=np.int8) % 5)
        .clamp_max(3)
        .to(torch.int64)
    )


# seq="ACDEFGHIKLMNPQRSTVWY"
def protein_tokenizer(seq):
    id = np.frombuffer(seq.encode(), dtype=np.int8) % 23
    return torch.from_numpy(id + (id < 5) - (id > 10) - (id > 16) + 3).to(torch.int64)


@torch.no_grad()
def data_collector(examples, outputs):
    results = dict()
    if "DNA" in outputs:
        results["DNA"] = torch.stack(
            [DNA_tokenizer(example["DNA"]) for example in examples]
        )

    if "protein" in outputs:
        results["protein"] = torch.stack(
            [protein_tokenizer(example["protein"]) for example in examples]
        )

    for label_name in BindTransformerConfig.label_names:
        if label_name in outputs:
            results[label_name] = torch.tensor(
                [example[label_name] for example in examples], dtype=torch.float32
            )

    return results


def train_validation_test_split(ds, validation_ratio, test_ratio, seed):
    ds = ds["train"].train_test_split(
        test_size=test_ratio + validation_ratio, seed=seed
    )
    ds2 = ds["test"].train_test_split(
        test_size=test_ratio / (test_ratio + validation_ratio),
        seed=seed,
    )
    ds["validation"] = ds2["train"]
    ds["test"] = ds2["test"]
    return ds
