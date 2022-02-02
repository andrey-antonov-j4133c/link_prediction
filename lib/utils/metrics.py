import pandas as pd
import numpy as np

from sklearn.metrics import *

METRICS = {
    "average_precision": average_precision_score,
    "roc_auc": roc_auc_score
}


def calculate_metrics(y_true: np.ndarray, y_score: np.ndarray, name: str, metrics: list) -> pd.Series:
    y_true = y_true.astype(int)
    data = {key: METRICS[key](y_true, y_score) for key in metrics if key in METRICS.keys()}
    return pd.Series(data, index=metrics, name=name)
