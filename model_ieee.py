"""Model training and evaluation. Edited by the agent.

The harness calls train_and_evaluate(). Return predictions for val and OOT.
GPU is auto-detected and used when available.
"""

import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.utils import resample

from harness.utils import get_gpu_info


def train_and_evaluate(X_train, y_train, X_val, y_val, X_oot, y_oot, config):
    """Train XGBoost + LightGBM ensemble. More models + diverse configs."""
    gpu = get_gpu_info()
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]

    all_val_preds = []
    all_oot_preds = []
    first_model = None

    # XGBoost ensemble (GPU) - 7 diverse models
    xgb_configs = [
        (3, 7, 0.03, 0.8, 0.8, 42),
        (10, 7, 0.03, 0.8, 0.8, 43),
        (30, 7, 0.03, 0.8, 0.8, 44),
        (5, 5, 0.05, 0.7, 0.7, 45),
        (10, 8, 0.02, 0.85, 0.85, 46),
        (15, 6, 0.04, 0.75, 0.75, 47),
        (20, 7, 0.025, 0.8, 0.7, 48),
    ]
    for ratio, depth, lr, subsamp, colsamp, seed in xgb_configs:
        n_neg = min(len(pos_idx) * ratio, len(neg_idx))
        neg_sample = resample(neg_idx, n_samples=n_neg, random_state=seed + ratio)
        idx = np.concatenate([pos_idx, neg_sample])

        model = xgb.XGBClassifier(
            n_estimators=1500,
            max_depth=depth,
            learning_rate=lr,
            subsample=subsamp,
            colsample_bytree=colsamp,
            scale_pos_weight=n_neg / len(pos_idx),
            tree_method=gpu["tree_method"],
            device=gpu["device"],
            eval_metric="aucpr",
            early_stopping_rounds=50,
            random_state=seed,
        )
        model.fit(
            X_train.iloc[idx], y_train[idx],
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        all_val_preds.append(model.predict_proba(X_val)[:, 1])
        all_oot_preds.append(model.predict_proba(X_oot)[:, 1])
        if first_model is None:
            first_model = model

    # LightGBM ensemble (CPU) - 5 diverse models
    lgb_configs = [
        (10, 7, 0.05, 0.8, 0.8, 100),
        (30, 6, 0.03, 0.85, 0.75, 101),
        (5, 8, 0.03, 0.75, 0.85, 102),
        (15, 7, 0.04, 0.8, 0.8, 103),
        (20, 6, 0.025, 0.8, 0.7, 104),
    ]
    for ratio, depth, lr, subsamp, colsamp, seed in lgb_configs:
        n_neg = min(len(pos_idx) * ratio, len(neg_idx))
        neg_sample = resample(neg_idx, n_samples=n_neg, random_state=seed + ratio)
        idx = np.concatenate([pos_idx, neg_sample])

        lgb_model = lgb.LGBMClassifier(
            n_estimators=1500,
            max_depth=depth,
            learning_rate=lr,
            subsample=subsamp,
            colsample_bytree=colsamp,
            scale_pos_weight=n_neg / len(pos_idx),
            device="cpu",
            min_child_samples=20,
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
            random_state=seed,
            verbose=-1,
        )
        lgb_model.fit(
            X_train.iloc[idx], y_train[idx],
            eval_set=[(X_val, y_val)],
        )
        all_val_preds.append(lgb_model.predict_proba(X_val)[:, 1])
        all_oot_preds.append(lgb_model.predict_proba(X_oot)[:, 1])

    # Average all predictions
    y_val_pred = np.mean(all_val_preds, axis=0)
    y_oot_pred = np.mean(all_oot_preds, axis=0)

    return {
        "y_val_pred": y_val_pred,
        "y_oot_pred": y_oot_pred,
        "model": first_model,
        "train_info": {
            "n_models": len(xgb_configs) + len(lgb_configs),
        },
    }
