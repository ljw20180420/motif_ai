import torch


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
    results["DNA_ids"] = DNA_tokenizer(
        [example["DNA"] for example in examples],
        [zinc_nums[example["index"]] for example in examples],
    )

    for key in examples[0].keys():
        if key == "DNA" or key == "index":
            continue

        results[key] = torch.tensor(
            [example[key] for example in examples],
            dtype=torch.float32,
        )

    return results


@torch.no_grad()
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
