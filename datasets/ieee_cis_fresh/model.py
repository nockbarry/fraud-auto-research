"""Baseline model for IEEE-CIS fresh track. Agent-editable."""
import xgboost as xgb
from harness.utils import get_gpu_info


def train_and_evaluate(X_train, y_train, X_val, y_val, X_oot, y_oot, config):
    gpu = get_gpu_info()
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    model = xgb.XGBClassifier(
        n_estimators=5000,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=15,
        reg_alpha=0.3,
        reg_lambda=3.0,
        gamma=0.3,
        scale_pos_weight=neg / max(pos, 1),
        tree_method=gpu["tree_method"],
        device=gpu["device"],
        eval_metric="aucpr",
        early_stopping_rounds=150,
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return {
        "y_val_pred": model.predict_proba(X_val)[:, 1],
        "y_oot_pred": model.predict_proba(X_oot)[:, 1],
        "model": model,
        "train_info": {"n_estimators_used": model.best_iteration, "device": gpu["device"]},
    }
