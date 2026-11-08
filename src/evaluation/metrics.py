"""Accuracy, QWK, per-class recall, confusion matrix. Owner: Maheen (Step 2, Week 9).

Every experiment (baseline, Focal Loss, SMOTE, hybrid) is scored by this one function,
so all numbers in the report are computed the same way.

Why these metrics:
- Accuracy: what [11] and [7] report, so we can put our number next to theirs.
- QWK (quadratic weighted kappa): the main metric for DR grading ([8], [10]). It punishes
  big mistakes (grade 0 predicted as 4) much more than small ones (grade 1 as 2).
- Per-class recall: accuracy hides the rare grades (75% of images are grade 0), so we
  report how many images of EACH grade are caught. Grade 1 (Mild) matters most (RQ3).
- Macro F1: average F1 over the 5 grades, every grade counts equally.
- Confusion matrix: shows which grades get mixed up with which.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, recall_score)

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]


def compute_metrics(y_true, y_pred, num_classes: int = 5) -> dict:
    """Return all metrics for one set of predictions (integer grades 0..num_classes-1)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    labels = list(range(num_classes))
    recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    # QWK is undefined when only one grade is present (smoke tests on tiny batches)
    if len(np.unique(np.concatenate([y_true, y_pred]))) < 2:
        qwk = 0.0
    else:
        qwk = cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "qwk": float(qwk),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class_recall": [float(r) for r in recall],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n": int(len(y_true)),
    }


def format_metrics(m: dict, class_names=CLASS_NAMES) -> str:
    """One readable block of text, for printing in notebooks."""
    lines = [f"n = {m['n']} | accuracy = {m['accuracy']:.4f} | QWK = {m['qwk']:.4f} | macro F1 = {m['macro_f1']:.4f}",
             "Recall per grade:"]
    for i, r in enumerate(m["per_class_recall"]):
        name = class_names[i] if i < len(class_names) else str(i)
        lines.append(f"  {i} {name:<14} {r:.3f}")
    return "\n".join(lines)


def plot_confusion_matrix(cm, class_names=CLASS_NAMES, title="Confusion matrix", save_path=None):
    """Counts plus row-normalised colour (each row = one true grade). Returns the figure."""
    import matplotlib.pyplot as plt
    cm = np.asarray(cm)
    row_sum = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum > 0)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=35, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted grade"); ax.set_ylabel("True grade"); ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, label="share of the true grade")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
