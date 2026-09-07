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


def best_threshold_at_fp(y_true: np.ndarray, score: np.ndarray,
                         fp_budget: float) -> float:
    """Largest predicted-positive set whose false-positive *rate* stays <=
    `fp_budget` (FP / actual-negatives).  Returns the score threshold.  This is
    how a bank operates a fraud model -- fix the tolerated rate of frozen
    legitimate customers, then maximise catches within it."""
    y_true = np.asarray(y_true)
    neg = float((y_true == 0).sum())
    if neg == 0:
        return 0.5
    order = np.argsort(-score, kind="stable")
    s = np.asarray(score)[order]
    y = y_true[order].astype(float)
    fp = np.cumsum(1.0 - y)
    ok = np.where(fp <= fp_budget * neg)[0]
    if len(ok) == 0:
        return float(s[0]) + abs(float(s[0])) * 1e-9 + 1e-12   # flag nothing
    L = int(ok[-1]) + 1                                        # flag the top L
    if L >= len(s):
        return float(s[-1]) - 1e-12
    # threshold between the L-th and (L+1)-th score; if they tie, push just above
    if s[L - 1] == s[L]:
        return float(s[L - 1]) + abs(float(s[L - 1])) * 1e-9 + 1e-12
    return float((s[L - 1] + s[L]) / 2.0)


def money_weighted_scores(y_true: np.ndarray, proba_pos: np.ndarray,
                          amount_at_risk: np.ndarray, fp_budget: float = 0.02,
                          threshold: float | None = None) -> dict:
    """Fraud-detection scores weighted by money protected.

    `money_recall` = £ at risk on caught mules / £ at risk on all mules -- the
    number a fraud-ops lead and a regulator actually ask for.  Evaluated at the
    threshold that maximises catches subject to a false-positive-rate budget.
    """
    y_true = np.asarray(y_true)
    amt = np.asarray(amount_at_risk, dtype=float)
    if threshold is None:
        threshold = best_threshold_at_fp(y_true, proba_pos, fp_budget)
    pred = (proba_pos >= threshold).astype(int)
    tp = ((pred == 1) & (y_true == 1))
    fp = int(((pred == 1) & (y_true == 0)).sum())
    neg = int((y_true == 0).sum())
    total_risk = amt[y_true == 1].sum()
    caught_risk = amt[tp].sum()
    plain = binary_scores(y_true, proba_pos, threshold)
    return {
        "money_recall": float(caught_risk / total_risk) if total_risk > 0 else 0.0,
        "money_at_risk_total": float(total_risk),
        "money_at_risk_caught": float(caught_risk),
        "fp_rate": float(fp / neg) if neg else 0.0,
        "recall": plain["recall"], "precision": plain["precision"],
        "f1": plain["f1"], "auc": plain["auc"],
        "threshold": float(threshold), "fp_budget": float(fp_budget),
    }


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
