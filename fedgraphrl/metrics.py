import numpy as np


def binary_scores(y_true: np.ndarray, proba_pos: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (proba_pos >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": (tp + tn) / max(len(y_true), 1),
        "auc": roc_auc(y_true, proba_pos),
    }


def best_f1_threshold(y_true: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """Vectorised search for the threshold maximising F1 (predict positive when
    score >= threshold).  Returns (best_f1, threshold)."""
    y_true = np.asarray(y_true)
    n_pos = int(y_true.sum())
    if len(y_true) == 0 or n_pos == 0:
        return 0.0, 0.5
    order = np.argsort(-score, kind="stable")
    y = y_true[order].astype(float)
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    prec = tp / np.maximum(tp + fp, 1.0)
    rec = tp / n_pos
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    i = int(np.argmax(f1))
    return float(f1[i]), float(score[order][i])


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    pos = score[y_true == 1]
    neg = score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    auc = (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)
