import torch
import numpy as np
from ..config import get_config

args = get_config(config_file="config_bind_transformer.ini")

outputs_train = ["seq", "bind"]
outputs_test = ["seq", "bind"]
outputs_inference = ["seq"]


@torch.no_grad()
def data_collector(examples, outputs):
    results = dict()
    if "seq" in outputs:
        results["seq"] = torch.stack(
            [
                torch.from_numpy(
                    np.frombuffer(example["seq"].encode(), dtype=np.int8) % 5
                )
                .clamp_max(3)
                .to(torch.int64)
                for example in examples
            ]
        )

    if "bind" in outputs:
        results["bind"] = torch.tensor(
            [example["bind"] for example in examples], dtype=torch.float32
        )

    return results


def train_validation_test_split(ds):
    ds = ds["train"].train_test_split(
        test_size=args.test_ratio + args.validation_ratio, seed=args.seed
    )
    ds2 = ds["test"].train_test_split(
        test_size=args.test_ratio / (args.test_ratio + args.validation_ratio),
        seed=args.seed,
    )
    ds["validation"] = ds2["train"]
    ds["test"] = ds2["test"]
    return ds
