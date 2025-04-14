import torch
from torch.nn.utils.rnn import pad_sequence
from .model import BindTransformerConfig

outputs_inference = [
    "protein",
    "second",
    "DNA",
]
outputs_train = outputs_inference + BindTransformerConfig.label_names
outputs_test = outputs_train

DNA_tokenmap = torch.zeros(91, dtype=torch.int64)
DNA_tokenmap[torch.frombuffer("XZACGT".encode(), dtype=torch.int8).to(torch.int64)] = (
    torch.arange(6)
)


def DNA_tokenizer(DNA):
    """
    DNA: XZACGT->012345。 X是mask token。Z是[CLS]token。ACGT是碱基token。
    """
    return DNA_tokenmap[
        torch.frombuffer(DNA.encode(), dtype=torch.int8).to(torch.int64)
    ]


protein_tokenmap = torch.zeros(90, dtype=torch.int64)
protein_tokenmap[
    torch.frombuffer("XACDEFGHIKLMNPQRSTVWY".encode(), dtype=torch.int8).to(torch.int64)
] = torch.arange(21)


def protein_tokenizer(protein):
    """
    protein: XACDEFGHIKLMNPQRSTVWY->0-20。X是mask token。ACDEFGHIKLMNPQRSTVWY是氨基酸。
    """
    return protein_tokenmap[
        torch.frombuffer(protein.encode(), dtype=torch.int8).to(torch.int64)
    ]


second_tokenmap = torch.zeros(91, dtype=torch.int64)
second_tokenmap[
    torch.frombuffer("XHBEGIPTS-KZ".encode(), dtype=torch.int8).to(torch.int64)
] = torch.arange(12)


def second_tokenizer(second):
    """
    second: XHBEGIPTS-KZ->0-11。X是mask token。HBEGIPTS-是二级结构。K是KRAB。Z是锌指。
    """
    return second_tokenmap[
        torch.frombuffer(second.encode(), dtype=torch.int8).to(torch.int64)
    ]


@torch.no_grad()
def data_collector(examples, proteins, seconds, outputs):
    results = dict()
    if "protein" in outputs:
        results["protein_ids"] = pad_sequence(
            [proteins[example["index"]] for example in examples],
            batch_first=True,
            padding_value=0,
        )
    if "second" in outputs:
        results["second_ids"] = pad_sequence(
            [seconds[example["index"]] for example in examples],
            batch_first=True,
            padding_value=0,
        )
    if "DNA" in outputs:
        results["DNA_ids"] = pad_sequence(
            [DNA_tokenizer("Z" + example["DNA"]) for example in examples],
            batch_first=True,
            padding_value=0,
        )

    for label_name in BindTransformerConfig.label_names:
        if label_name in outputs:
            results[label_name] = torch.tensor(
                [example[label_name] for example in examples], dtype=torch.int64
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
