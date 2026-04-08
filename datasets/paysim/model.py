"""Model training for PaySim mobile money fraud dataset. Edited by the agent.

The harness calls train_and_evaluate(). Return predictions for val and OOT.
GPU is auto-detected and used when available.

NOTE: PaySim has a significant class imbalance (~0.13% overall fraud rate,
but fraud rate varies by type — TRANSFER: 0.77%, CASH_OUT: 0.18%).
scale_pos_weight based on train class ratio is used for imbalance handling.
eval_metric="aucpr" is critical for imbalanced datasets.
"""

import numpy as np
import xgboost as xgb

from harness.utils import get_gpu_info


def train_and_evaluate(X_train, y_train, X_val, y_val, X_oot, y_oot, config):
    """Train model and return predictions."""
    gpu = get_gpu_info()

    # Dynamic scale_pos_weight from train set
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = neg / max(pos, 1)

    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        min_child_weight=10,
        reg_alpha=0.1,
        reg_lambda=1.0,
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
            "scale_pos_weight": round(spw, 2),
            "device": gpu["device"],
        },
    }
