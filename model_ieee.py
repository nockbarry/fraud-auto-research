"""Baseline model for IEEE-CIS dataset. Agent-editable."""
import xgboost as xgb
from harness.utils import get_gpu_info


def train_and_evaluate(X_train, y_train, X_val, y_val, X_oot, y_oot, config):
    gpu = get_gpu_info()
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    model = xgb.XGBClassifier(
        n_estimators=2000,
        max_depth=4,           # reduced from 6 to reduce overfit (train_val_psi was 0.29)
        learning_rate=0.03,    # lower LR with more trees
        subsample=0.80,        # row subsampling
        colsample_bytree=0.70, # feature subsampling per tree
        min_child_weight=10,   # fewer splits in small nodes (reduces memorization)
        gamma=1.0,             # min gain to split
        reg_lambda=5.0,        # L2 regularization
        scale_pos_weight=neg / max(pos, 1),
        tree_method=gpu["tree_method"],
        device=gpu["device"],
        eval_metric="aucpr",
        early_stopping_rounds=50,
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return {
        "y_val_pred": model.predict_proba(X_val)[:, 1],
        "y_oot_pred": model.predict_proba(X_oot)[:, 1],
        "model": model,
        "train_info": {"n_estimators_used": model.best_iteration, "device": gpu["device"]},
    }
