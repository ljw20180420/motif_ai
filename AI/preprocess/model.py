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

    @classmethod
    def _get_feature(
        cls,
        input: dict | None,
        label: dict | None,
    ) -> tuple[np.ndarray]:
        assert input is not None or label is not None, (
            "at least one of input and label should not be None"
        )
        if input is not None:
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

        if input is not None and label is not None:
            return X_value, y_value
        elif input is not None:
            return X_value
        return y_value

    @classmethod
    def _get_feature_all(
        cls,
        dataloader: torch.utils.data.DataLoader,
        my_generator: MyGenerator,
        output_X: bool,
        output_y: bool,
    ) -> tuple[np.ndarray]:
        assert output_X or output_y, "output nothing is not allowed"
        if output_X:
            X = []
        if output_y:
            y = []
        for examples in tqdm(dataloader):
            batch = cls.data_collator(
                examples, output_label=output_y, my_generator=my_generator
            )
            values = cls._get_feature(
                input=batch["input"] if output_X else None,
                label=batch["label"] if output_y else None,
            )
            if output_X and output_y:
                X_value, y_value = values
            elif output_X:
                X_value = values
            else:
                y_value = values
            if output_X:
                X.append(X_value.astype(np.int8))
            if output_y:
                y.append(y_value.astype(np.int8))

        if output_X:
            X = np.concatenate(X)
        if output_y:
            y = np.concatenate(y)

        if output_X and output_y:
            return X, y
        elif output_X:
            return X
        return y

    @classmethod
    def hpo(cls, trial: optuna.Trial, cfg: jsonargparse.Namespace) -> None:
        pass
