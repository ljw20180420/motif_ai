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
    fp,
):
    results = {}
    if protein_tokenizer:
        results["protein_ids"] = protein_tokenizer(
            [proteins[example["index"]] for example in examples]
        )
    if second_tokenizer:
        results["second_ids"] = second_tokenizer(
            [seconds[example["index"]] for example in examples],
        )
    if DNA_tokenizer:
        results["DNA_ids"] = DNA_tokenizer(
            [example["DNA"] for example in examples],
            [zinc_nums[example["index"]] for example in examples],
        )

    dtype = {"fp16": torch.float16, "fp32": torch.float32, "fp64": torch.float64}[fp]

    for key in examples[0].keys():
        if key is "DNA" or key is "index":
            continue

        results[key] = torch.tensor(
            [example[key] for example in examples],
            dtype=dtype,
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
