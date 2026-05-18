import jsonargparse
import numpy as np
import optuna
import torch
from common_ai.generator import MyGenerator
from common_ai.initializer import MyInitializer
from common_ai.model import MyModelAbstract
from tqdm import tqdm

from .data_collator import DataCollator


class MLBase(MyModelAbstract):
    def my_initialize_model(
        self, my_initializer: MyInitializer, my_generator: MyGenerator
    ) -> None:
        pass

    @classmethod
    def _get_feature(
        cls,
        input: dict,
        label: dict | None,
    ) -> tuple[np.ndarray]:
        X_value = np.concatenate(
            (
                input["dna_id"].cpu().numpy(),
                input["protein_id"].cpu().numpy(),
                input["second_id"].cpu().numpy(),
            ),
            axis=1,
        )

        if label is not None:
            y_value = label["bind"].cpu().numpy()
            return X_value, y_value

        return X_value

    def _get_feature_all(
        self,
        dataloader: torch.utils.data.DataLoader,
        my_generator: MyGenerator,
        output_label: bool,
    ) -> tuple[np.ndarray]:
        X = []
        if output_label:
            y = []
        for examples in tqdm(dataloader):
            batch = self.data_collator(
                examples, output_label=output_label, my_generator=my_generator
            )
            if output_label:
                X_value, y_value = self._get_feature(
                    input=batch["input"], label=batch["label"]
                )
            else:
                X_value = self._get_feature(input=batch["input"], label=None)
            X.append(X_value.astype(np.int8))
            if output_label:
                y.append(y_value.astype(np.int8))

        X = np.concatenate(X)
        y = np.concatenate(y)

        if output_label:
            return X, y

        return X

    @classmethod
    def hpo(cls, trial: optuna.Trial, cfg: jsonargparse.Namespace) -> None:
        pass
