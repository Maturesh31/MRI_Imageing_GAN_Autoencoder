import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def compute_all_metrics(y_true, y_prob, threshold=0.5):
    """
    Computes all standard evaluation metrics for binary classification:
    - Accuracy: (TP + TN) / Total
    - Precision: TP / (TP + FP)
    - Recall (Sensitivity): TP / (TP + FN)
    - Specificity: TN / (TN + FP)
    - F1-Score: 2 * Precision * Recall / (Precision + Recall)
    - AUC-ROC: Area under ROC curve
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.5

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        spec = 0.0

    metrics = {
        "accuracy": round(acc, 4),
        "accuracy_pct": round(acc * 100, 2),
        "precision": round(prec, 4),
        "recall_sensitivity": round(rec, 4),
        "specificity": round(spec, 4),
        "f1_score": round(f1, 4),
        "auc_roc": round(auc, 4),
        "confusion_matrix": {
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn)
        }
    }
    return metrics

def print_metrics_table(exp_name, metrics):
    """Pretty print metric results."""
    print("=" * 60)
    print(f" EXPERIMENT RESULTS: {exp_name}")
    print("=" * 60)
    print(f" Accuracy        : {metrics['accuracy_pct']}% ({metrics['accuracy']:.4f})")
    print(f" Precision       : {metrics['precision']:.4f}")
    print(f" Recall / Sens   : {metrics['recall_sensitivity']:.4f}")
    print(f" Specificity     : {metrics['specificity']:.4f}")
    print(f" F1-Score        : {metrics['f1_score']:.4f}")
    print(f" AUC-ROC         : {metrics['auc_roc']:.4f}")
    cm = metrics["confusion_matrix"]
    print(f" Confusion Matrix: TP={cm['TP']}, TN={cm['TN']}, FP={cm['FP']}, FN={cm['FN']}")
    print("=" * 60)
