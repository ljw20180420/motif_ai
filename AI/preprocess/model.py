from typing import Optional

import jsonargparse
import numpy as np
import optuna
import torch
from common_ai.generator import MyGenerator
from common_ai.initializer import MyInitializer
from common_ai.model import MyModelAbstract
from tqdm import tqdm


class MLBase(MyModelAbstract):
    def my_initialize_model(
        self, my_initializer: MyInitializer, my_generator: MyGenerator
    ) -> None:
        pass

    def _get_feature(
        self,
        input: dict,
        label: Optional[dict],
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
        self, dataloader: torch.utils.data.DataLoader, my_generator: MyGenerator
    ) -> tuple[np.ndarray]:
        X, y = [], []
        for examples in tqdm(dataloader):
            batch = self.data_collator(
                examples, output_label=True, my_generator=my_generator
            )
            X_value, y_value = self._get_feature(
                input=batch["input"], label=batch["label"]
            )
            X.append(X_value)
            y.append(y_value)

        X = np.concatenate(X)
        y = np.concatenate(y)

        return X, y

    @classmethod
    def hpo(cls, trial: optuna.Trial, cfg: jsonargparse.Namespace) -> None:
        pass
