"""FDH model. Agent-editable."""
import xgboost as xgb
from harness.utils import get_gpu_info


def train_and_evaluate(X_train, y_train, X_val, y_val, X_oot, y_oot, config):
    gpu = get_gpu_info()
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
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
