"""Model training for FDH dataset. Edited by the agent.

GPU is auto-detected and used when available.
FDH has 0.84% fraud rate — scale_pos_weight from train ratio.
"""

import numpy as np
import xgboost as xgb

from harness.utils import get_gpu_info


def train_and_evaluate(X_train, y_train, X_val, y_val, X_oot, y_oot, config):
    """Train model and return predictions."""
    gpu = get_gpu_info()

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = neg / max(pos, 1)

    model = xgb.XGBClassifier(
        n_estimators=1500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.75,
        colsample_bytree=0.75,
        scale_pos_weight=spw,
        min_child_weight=20,
        reg_alpha=0.5,
        reg_lambda=2.0,
        gamma=0.1,
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
