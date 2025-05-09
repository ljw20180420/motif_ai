import torch
import subprocess
import pandas as pd
from io import StringIO
import os


@torch.no_grad()
def data_collector(
    examples,
    proteins,
    seconds,
    zinc_nums,
    DNA_tokenizer,
    protein_tokenizer,
    second_tokenizer,
):
    results = {}
    results["protein_ids"] = protein_tokenizer(
        [proteins[example["index"]] for example in examples]
    )
    results["second_ids"] = second_tokenizer(
        [seconds[example["index"]] for example in examples],
    )
    results["dna_ids"] = DNA_tokenizer(
        [example["dna"] for example in examples],
        [zinc_nums[example["index"]] for example in examples],
    )

    for key in examples[0].keys():
        if key == "dna" or key == "index":
            continue

        results[key] = torch.tensor(
            [example[key] for example in examples],
            dtype=torch.float32,
        )

    return results


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
