import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from common_ai.generator import MyGenerator
from common_ai.optimizer import MyOptimizer
from common_ai.profiler import MyProfiler
from common_ai.train import MyTrain
from tqdm import tqdm

from ..data_collator import DataCollator
from ..model import MLBase


class LightSeq(lgb.Sequence):
    def __init__(
        self,
        dataloader: torch.utils.data.DataLoader,
        data_collator: DataCollator,
        my_generator: MyGenerator,
    ) -> None:
        self.dataset = dataloader.dataset
        self.data_collator = data_collator
        self.my_generator = my_generator
        self.batch_size = self.data_collator.batch_size

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int | slice) -> np.ndarray:
        if isinstance(idx, int):
            examples = [self.dataset[idx]]
        elif isinstance(idx, slice):
            examples = [
                self.dataset[i]
                for i in range(start=idx.start, stop=idx.stop, step=idx.step)
            ]
        batch = self.data_collator(
            examples, output_label=False, my_generator=self.my_generator
        )
        X = MLBase._get_feature(input=batch["input"], label=None)
        if isinstance(idx, int):
            X = X[0]

        return X.astype(np.int8)


class LightGBM(MLBase):
    def __init__(
        self,
        protein_feature: os.PathLike,
        protein_length: int,
        dna_length: int,
        subsample: float,
        colsample_bynode: float,
        eta: float,
        num_boost_round: int,
    ) -> None:
        """LigtGBM arguments.

        Args:
            protein_feature: file contains info for mouse C2H2 zinc fingers.
            protein_length: maximally allowed protein length.
            dna_length: maximally allowed DNA length.
            subsample: subsample ratio of the training instances.
            colsample_bynode: subsample ratio of columns for each node (split).
            eta: Shrink of step size after each round.
            num_boost_round: Number of trees generated in single epochs.
        """
        self.subsample = subsample
        self.colsample_bynode = colsample_bynode
        self.eta = eta
        self.num_boost_round = num_boost_round

        self.data_collator = DataCollator(protein_feature, protein_length, dna_length)

        self.booster = None

    def eval_output(
        self, examples: list[dict], batch: dict, my_generator: MyGenerator
    ) -> pd.DataFrame:
        X_value = self._get_feature(
            input=batch["input"],
            label=None,
        )
        batch_size = X_value.shape[0]
        probas = self.booster.predict(data=X_value)
        df = pd.DataFrame({
            "sample_idx": np.arange(batch_size),
            "proba": probas,
            "DNA": [example["DNA"] for example in examples],
            "protein": [example["protein"] for example in examples],
        })

        return df

    def state_dict(self) -> dict:
        return {
            "booster": torch.frombuffer(
                bytearray(self.booster.model_to_string().encode()),
                dtype=torch.uint8,
            )
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.booster = lgb.Booster(
            model_str=state_dict["booster"].numpy().tobytes().decode()
        )

    def _train_booster(self, my_generator: MyGenerator) -> dict:
        eval_result = {}
        self.booster = lgb.train(
            params={
                "subsample": self.subsample,
                "colsample_bynode": self.colsample_bynode,
                "eta": self.eta,
                "objective": "binary",
                "seed": my_generator.seed,
                "device": self.device,
            },
            train_set=self.train_data,
            num_boost_round=self.num_boost_round,
            valid_sets=[self.train_data, self.eval_data],
            valid_names=["train", "eval"],
            init_model=self.booster,
            keep_training_booster=True,
            callbacks=[lgb.record_evaluation(eval_result)],
        )

        return eval_result

    def my_train_epoch(
        self,
        my_train: MyTrain,
        train_dataloader: torch.utils.data.DataLoader,
        eval_dataloader: torch.utils.data.DataLoader,
        my_generator: MyGenerator,
        my_optimizer: MyOptimizer,
        my_profiler: MyProfiler,
        metrics: dict,
    ) -> tuple:
        if not hasattr(self, "train_data") or not hasattr(self, "eval_data"):
            X_eval, y_eval = self._get_feature_all(
                dataloader=eval_dataloader,
                data_collator=self.data_collator,
                my_generator=my_generator,
                output_label=True,
            )
            light_seq = LightSeq(
                dataloader=train_dataloader,
                data_collator=self.data_collator,
                my_generator=my_generator,
            )
            self.train_data = lgb.Dataset(
                data=light_seq,
                label=np.array(light_seq.dataset["bind"], dtype=np.int8),
                categorical_feature=list(range(X_eval.shape[-1])),
            )
            self.eval_data = lgb.Dataset(
                data=X_eval,
                label=y_eval,
                reference=self.train_data,
                categorical_feature=list(range(X_eval.shape[-1])),
            )
            del X_eval
            del y_eval

        eval_result = self._train_booster(my_generator)

        return (
            np.mean(eval_result["train"]["binary_logloss"]).item()
            * self.train_data.num_data(),
            self.train_data.num_data(),
            float("nan"),
        )

    def my_eval_epoch(
        self,
        my_train: MyTrain,
        eval_dataloader: torch.utils.data.DataLoader,
        my_generator: MyGenerator,
        metrics: dict,
    ) -> tuple:
        eval_loss = (
            self.booster.eval(data=self.eval_data, name="eval")[0][2].item()
            * self.eval_data.num_data()
        )
        for examples in tqdm(eval_dataloader):
            batch = self.data_collator(
                examples, output_label=True, my_generator=my_generator
            )
            df = self.eval_output(examples, batch, my_generator)
            for metric_name, metric_fun in metrics.items():
                metric_fun.step(
                    df=df,
                    examples=examples,
                    batch=batch,
                )

        metric_loss_dict = {}
        for metric_name, metric_fun in metrics.items():
            metric_loss_dict[metric_name] = metric_fun.epoch()

        return eval_loss, self.eval_data.num_data(), metric_loss_dict
