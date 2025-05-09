import numpy as np
import evaluate
import sklearn
import scipy
from huggingface_hub import HfFileSystem


def download_metrics():
    fs = HfFileSystem()
    for metric in [
        "accuracy",
        "recall",
        "precision",
        "f1",
        "matthews_correlation",
        "confusion_matrix",
        "roc_auc",
        "brier_score",
    ]:
        with fs.open(f"spaces/evaluate-metric/{metric}/{metric}.py", "rb") as rd, open(
            f"bind_transformer/metrics/{metric}.py", "wb"
        ) as ld:
            ld.write(rd.read())


def select_threshold(values, bind, proba=True):
    if proba:
        min_val = 0
        max_val = 1
    else:
        min_val = min(values)
        max_val = max(values)
    metric = evaluate.load("bind_transformer/metrics/accuracy.py")
    best_result = {"accuracy": -1}
    for thres in np.linspace(min_val, max_val, 99):
        pred = values >= thres
        result = metric.compute(predictions=pred, references=bind)
        if result["accuracy"] > best_result["accuracy"]:
            best_result = result
            best_thres = thres
    return best_thres


def hard_metric(pred, y):
    metrics = evaluate.combine(
        [
            "bind_transformer/metrics/f1.py",
            "bind_transformer/metrics/accuracy.py",
            "bind_transformer/metrics/recall.py",
            "bind_transformer/metrics/precision.py",
            "bind_transformer/metrics/matthews_correlation.py",
            "bind_transformer/metrics/confusion_matrix.py",
        ]
    )
    results = metrics.compute(predictions=pred, references=y)
    confusion_matrix = results.pop("confusion_matrix")
    return {
        **results,
        "true_negative": confusion_matrix[0, 0],
        "false_positive": confusion_matrix[0, 1],
        "false_negative": confusion_matrix[1, 0],
        "true_positive": confusion_matrix[1, 1],
    }


def compute_metrics_probabilities(bind_probabilities: np.ndarray, binds: np.ndarray):
    best_thres = select_threshold(bind_probabilities, binds)

    hard_results = hard_metric(bind_probabilities >= best_thres, binds)

    roc_auc_metrics = evaluate.load("bind_transformer/metrics/roc_auc.py")
    roc_auc_results = roc_auc_metrics.compute(
        prediction_scores=bind_probabilities, references=binds
    )

    # huggingface evaluate官方没有pr auc，只能用scikit-learn的average_precision_score
    pr_auc_results = {
        "pr_auc": sklearn.metrics.average_precision_score(binds, bind_probabilities)
    }

    brier_score_metrics = evaluate.load("bind_transformer/metrics/brier_score.py")
    brier_score_results = brier_score_metrics.compute(
        predictions=bind_probabilities, references=binds
    )

    return {
        "threshold": best_thres,
        **hard_results,
        **roc_auc_results,
        **pr_auc_results,
        **brier_score_results,
    }


def compute_metrics(logits: np.ndarray, binds: np.ndarray):
    bind_probabilities = scipy.special.expit(logits)
    return compute_metrics_probabilities(bind_probabilities, binds)


# from evaluate.visualization import radar_plot

# plot = radar_plot(data=results, model_names=models, invert_range=["latency_in_seconds"])
# plot.show()
# plot.savefig(bbox_inches="tight")
