"""Model training and evaluation. Edited by the agent.

The harness calls train_and_evaluate(). Return predictions for val and OOT.
"""

import numpy as np
import xgboost as xgb


def train_and_evaluate(X_train, y_train, X_val, y_val, X_oot, y_oot, config):
    """Train model and return predictions.

    Args:
        X_train, y_train: Training features and labels.
        X_val, y_val: Validation features and labels (used for early stopping).
        X_oot, y_oot: Out-of-time features and labels (used for final evaluation).
        config: Loaded config.yaml as dict.

    Returns:
        dict with keys:
            y_val_pred: np.ndarray of predicted probabilities on validation set
            y_oot_pred: np.ndarray of predicted probabilities on OOT set
            model: trained model object
            train_info: dict with training metadata
    """
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    n_features = X_train.shape[1]

    # Adaptive model params based on feature count
    if n_features > 60:
        max_depth, colsample, n_est, lr = 8, 0.6, 1500, 0.03
    else:
        max_depth, colsample, n_est, lr = 6, 0.8, 1000, 0.05

    # Cap class weight to avoid over-flagging
    capped_weight = min(scale_pos_weight, 50.0)

    model = xgb.XGBClassifier(
        n_estimators=n_est,
        max_depth=max_depth,
        learning_rate=lr,
        subsample=0.8,
        colsample_bytree=colsample,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=capped_weight,
        eval_metric="aucpr",
        early_stopping_rounds=80,
        random_state=42,
    )

    # Train primary model on full data
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Subsampled ensemble: train 2 additional models on undersampled data
    from sklearn.utils import resample

    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    models = [model]

    for ratio in [3, 10, 30]:
        n_neg = min(len(pos_idx) * ratio, len(neg_idx))
        neg_sample = resample(neg_idx, n_samples=n_neg, replace=False, random_state=42 + ratio)
        idx = np.concatenate([pos_idx, neg_sample])
        np.random.RandomState(ratio).shuffle(idx)

        sub_model = xgb.XGBClassifier(
            n_estimators=500, max_depth=max_depth, learning_rate=0.05,
            subsample=0.8, colsample_bytree=colsample,
            min_child_weight=3, eval_metric="aucpr",
            early_stopping_rounds=50, random_state=42 + ratio,
        )
        sub_model.fit(X_train.iloc[idx], y_train[idx], eval_set=[(X_val, y_val)], verbose=False)
        models.append(sub_model)

    # Rank-average predictions
    y_val_pred = np.mean([m.predict_proba(X_val)[:, 1] for m in models], axis=0)
    y_oot_pred = np.mean([m.predict_proba(X_oot)[:, 1] for m in models], axis=0)

    return {
        "y_val_pred": y_val_pred,
        "y_oot_pred": y_oot_pred,
        "model": model,
        "train_info": {
            "n_estimators_used": model.best_iteration,
            "scale_pos_weight": round(capped_weight, 2),
            "n_ensemble_models": len(models),
        },
    }
