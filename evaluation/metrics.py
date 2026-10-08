"""Metrik dan uji statistik untuk evaluasi Dataset Research."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
import numpy as np
from scipy.stats import binomtest, rankdata, wilcoxon


def classification_metrics(y_true: Sequence, y_pred: Sequence) -> dict:
    """Hitung accuracy, macro precision/recall/F1, dan confusion matrix."""
    truth = list(y_true)
    pred = list(y_pred)
    if not truth or len(truth) != len(pred):
        raise ValueError("y_true dan y_pred harus memiliki panjang sama dan tidak kosong.")
    labels = sorted(set(truth) | set(pred), key=str)
    matrix = [[0 for _ in labels] for _ in labels]
    positions = {label: index for index, label in enumerate(labels)}
    for actual, predicted in zip(truth, pred):
        matrix[positions[actual]][positions[predicted]] += 1
    precisions, recalls, f1s = [], [], []
    for index in range(len(labels)):
        tp = matrix[index][index]
        fp = sum(matrix[row][index] for row in range(len(labels))) - tp
        fn = sum(matrix[index]) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "n": len(truth),
        "labels": labels,
        "accuracy": sum(a == p for a, p in zip(truth, pred)) / len(truth),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "confusion_matrix": matrix,
    }


def precision_at_k(relevance: Sequence[float], k: int) -> float:
    """Hitung precision@k; setiap nilai relevansi positif dianggap relevan."""
    values = list(relevance)
    if k <= 0:
        raise ValueError("k harus lebih besar dari nol.")
    if not values:
        return 0.0
    selected = values[:k]
    return sum(value > 0 for value in selected) / k


def recall_at_k(relevance: Sequence[float], k: int) -> float:
    """Hitung recall@k terhadap seluruh item relevan dalam daftar yang dinilai."""
    values = list(relevance)
    if k <= 0:
        raise ValueError("k harus lebih besar dari nol.")
    total = sum(value > 0 for value in values)
    return sum(value > 0 for value in values[:k]) / total if total else 0.0


def mean_reciprocal_rank(relevance_lists: Sequence[Sequence[float]]) -> float:
    """Hitung MRR; item relevan pertama memiliki nilai relevansi positif."""
    if not relevance_lists:
        return 0.0
    reciprocal_ranks = []
    for values in relevance_lists:
        rank = next((i for i, value in enumerate(values, start=1) if value > 0), None)
        reciprocal_ranks.append(1 / rank if rank else 0.0)
    return float(np.mean(reciprocal_ranks))


def ndcg_at_k(relevance: Sequence[float], k: int, ideal_relevance: Sequence[float] | None = None) -> float:
    """Hitung nDCG@k memakai gain eksponensial dan log2 discount."""
    values = list(relevance)
    if k <= 0:
        raise ValueError("k harus lebih besar dari nol.")
    gains = [(2 ** max(float(value), 0.0) - 1) / np.log2(index + 2) for index, value in enumerate(values[:k])]
    ideal_values = values if ideal_relevance is None else list(ideal_relevance)
    ideal = sorted((max(float(value), 0.0) for value in ideal_values), reverse=True)[:k]
    ideal_dcg = sum((2 ** value - 1) / np.log2(index + 2) for index, value in enumerate(ideal))
    return float(sum(gains) / ideal_dcg) if ideal_dcg else 0.0


def cohens_kappa(rater_a: Sequence, rater_b: Sequence) -> float:
    """Hitung Cohen's kappa untuk dua daftar label kategorikal berpasangan."""
    a, b = list(rater_a), list(rater_b)
    if not a or len(a) != len(b):
        raise ValueError("Dua daftar penilai harus memiliki panjang sama dan tidak kosong.")
    observed = sum(x == y for x, y in zip(a, b)) / len(a)
    labels = set(a) | set(b)
    expected = sum(Counter(a)[label] * Counter(b)[label] for label in labels) / (len(a) ** 2)
    return 1.0 if expected == 1.0 and observed == 1.0 else (observed - expected) / (1 - expected) if expected < 1 else 0.0


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict:
    """Hitung CI percentile bootstrap yang deterministik dengan seed tetap."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("values harus berupa daftar angka finite dan tidak kosong.")
    if not 0 < confidence < 1 or n_resamples < 1:
        raise ValueError("confidence harus di antara 0 dan 1 dan n_resamples positif.")
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        samples[index] = statistic(rng.choice(array, size=len(array), replace=True))
    alpha = 1 - confidence
    return {
        "estimate": float(statistic(array)),
        "lower": float(np.quantile(samples, alpha / 2)),
        "upper": float(np.quantile(samples, 1 - alpha / 2)),
        "confidence": confidence,
        "n_resamples": n_resamples,
        "seed": seed,
    }


def mcnemar_test(y_true: Sequence, prediction_a: Sequence, prediction_b: Sequence) -> dict:
    """Uji McNemar exact untuk dua model pada observasi yang sama."""
    truth, pred_a, pred_b = map(list, (y_true, prediction_a, prediction_b))
    if not truth or len(truth) != len(pred_a) or len(truth) != len(pred_b):
        raise ValueError("Label dan prediksi harus memiliki panjang sama dan tidak kosong.")
    a_only = sum((a == y) and (b != y) for y, a, b in zip(truth, pred_a, pred_b))
    b_only = sum((b == y) and (a != y) for y, a, b in zip(truth, pred_a, pred_b))
    discordant = a_only + b_only
    p_value = float(binomtest(min(a_only, b_only), discordant, 0.5, alternative="two-sided").pvalue) if discordant else 1.0
    return {"a_correct_b_wrong": a_only, "b_correct_a_wrong": b_only,
            "discordant_pairs": discordant, "p_value": p_value,
            "effect_accuracy_difference": (a_only - b_only) / len(truth)}


def wilcoxon_signed_rank(values_a: Sequence[float], values_b: Sequence[float]) -> dict:
    """Uji Wilcoxon berpasangan dan rank-biserial effect size."""
    a, b = np.asarray(values_a, dtype=float), np.asarray(values_b, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or len(a) != len(b) or not len(a):
        raise ValueError("Dua daftar skor harus memiliki panjang sama dan tidak kosong.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Skor harus finite.")
    differences = a - b
    nonzero = differences != 0
    if not nonzero.any():
        return {"statistic": 0.0, "p_value": 1.0, "median_difference": 0.0, "rank_biserial": 0.0, "n_pairs": len(a)}
    result = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox", method="auto")
    ranks = rankdata(np.abs(differences[nonzero]))
    positive = float(ranks[differences[nonzero] > 0].sum())
    negative = float(ranks[differences[nonzero] < 0].sum())
    denominator = positive + negative
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue),
            "median_difference": float(np.median(differences)),
            "rank_biserial": (positive - negative) / denominator if denominator else 0.0,
            "n_pairs": len(a)}


def majority_baseline(y_train: Sequence, n_predictions: int) -> list:
    """Hasil baseline klasifikasi yang selalu memilih kelas mayoritas train."""
    values = list(y_train)
    if not values or n_predictions < 0:
        raise ValueError("y_train harus berisi label dan n_predictions tidak negatif.")
    counts = Counter(values)
    majority = max(counts, key=lambda label: (counts[label], str(label)))
    return [majority] * n_predictions


def random_ranking_baseline(n_items: int, seed: int = 42) -> list[int]:
    """Buat peringkat acak deterministik untuk pembanding retrieval."""
    if n_items < 0:
        raise ValueError("n_items tidak boleh negatif.")
    return np.random.default_rng(seed).permutation(n_items).tolist()
