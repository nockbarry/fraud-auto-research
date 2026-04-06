"""Model training and evaluation. Edited by the agent.
GPU is auto-detected and used when available.
"""

import numpy as np
import xgboost as xgb
from sklearn.utils import resample

from harness.utils import get_gpu_info


def train_and_evaluate(X_train, y_train, X_val, y_val, X_oot, y_oot, config):
    """Train ensemble model and return predictions."""
    gpu = get_gpu_info()
    pos = y_train.sum()
    neg = len(y_train) - pos
    raw_weight = neg / pos if pos > 0 else 1.0
    scale_pos_weight = min(raw_weight, 1)  # No class weighting

    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]

    models = []
    # Full-data models with different depths and seeds
    for depth in [3, 4, 5, 6, 7, 8]:
        for seed_offset in [0, 100, 200, 300]:
            model = xgb.XGBClassifier(
                n_estimators=1000,
                max_depth=depth,
                learning_rate=0.02,
                scale_pos_weight=scale_pos_weight,
                subsample=0.8,
                colsample_bytree=0.4,
                min_child_weight=5,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                tree_method=gpu["tree_method"],
                device=gpu["device"],
                eval_metric="aucpr",
                early_stopping_rounds=50,
                random_state=42 + depth + seed_offset,
            )
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            models.append(model)

    # Subsampled models
    for ratio in [5, 10, 30]:
        n_neg = min(len(pos_idx) * ratio, len(neg_idx))
        neg_sample = resample(neg_idx, n_samples=n_neg, random_state=42 + ratio)
        idx = np.concatenate([pos_idx, neg_sample])

        model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            tree_method=gpu["tree_method"],
            device=gpu["device"],
            eval_metric="aucpr",
            early_stopping_rounds=50,
            random_state=42 + ratio,
        )
        model.fit(X_train.iloc[idx], y_train[idx], eval_set=[(X_val, y_val)], verbose=False)
        models.append(model)

    # Rank-average predictions
    from scipy.stats import rankdata
    val_ranks = []
    oot_ranks = []
    for m in models:
        vp = m.predict_proba(X_val)[:, 1]
        op = m.predict_proba(X_oot)[:, 1]
        val_ranks.append(rankdata(vp) / len(vp))
        oot_ranks.append(rankdata(op) / len(op))

    y_val_pred = np.mean(val_ranks, axis=0)
    y_oot_pred = np.mean(oot_ranks, axis=0)

    return {
        "y_val_pred": y_val_pred,
        "y_oot_pred": y_oot_pred,
        "model": models[0],
        "train_info": {
            "n_models": len(models),
            "scale_pos_weight": round(scale_pos_weight, 2),
        },
    }
