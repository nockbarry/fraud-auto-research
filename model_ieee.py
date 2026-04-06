"""Model training for IEEE-CIS dataset. Edited by the agent.

The harness calls train_and_evaluate(). Return predictions for val and OOT.
GPU is auto-detected and used when available.
"""

import numpy as np
import xgboost as xgb

from harness.utils import get_gpu_info


def train_and_evaluate(X_train, y_train, X_val, y_val, X_oot, y_oot, config):
    """Train model and return predictions."""
    gpu = get_gpu_info()
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        tree_method=gpu["tree_method"],
        device=gpu["device"],
        eval_metric="aucpr",
        early_stopping_rounds=50,
        random_state=42,
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_val_pred = model.predict_proba(X_val)[:, 1]
    y_oot_pred = model.predict_proba(X_oot)[:, 1]

    return {
        "y_val_pred": y_val_pred,
        "y_oot_pred": y_oot_pred,
        "model": model,
        "train_info": {
            "n_estimators_used": model.best_iteration,
            "scale_pos_weight": round(scale_pos_weight, 2),
            "device": gpu["device"],
        },
    }
