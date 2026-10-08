import pytest

from evaluation.metrics import (
    bootstrap_ci,
    classification_metrics,
    cohens_kappa,
    mean_reciprocal_rank,
    mcnemar_test,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    wilcoxon_signed_rank,
)


def test_classification_metrics_match_hand_calculation():
    result = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["macro_precision"] == pytest.approx(5 / 6)
    assert result["macro_recall"] == pytest.approx(0.75)
    assert result["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]


def test_ranking_metrics_match_hand_calculation():
    relevance = [3, 0, 2, 1]
    assert precision_at_k(relevance, 2) == pytest.approx(0.5)
    assert recall_at_k(relevance, 2) == pytest.approx(1 / 3)
    assert mean_reciprocal_rank([[0, 1], [0, 0], [3]]) == pytest.approx(0.5)
    assert ndcg_at_k([3, 2], 2) == pytest.approx(1.0)
    assert ndcg_at_k([0, 2, 3], 2) < 1.0


def test_cohens_kappa_matches_hand_calculation():
    assert cohens_kappa(["a", "a", "b", "b"], ["a", "b", "b", "b"]) == pytest.approx(0.5)


def test_bootstrap_ci_is_reproducible():
    first = bootstrap_ci([0, 1, 1, 1], n_resamples=300, seed=17)
    second = bootstrap_ci([0, 1, 1, 1], n_resamples=300, seed=17)
    assert first == second
    assert first["lower"] <= first["estimate"] <= first["upper"]


def test_mcnemar_and_wilcoxon_return_p_value_and_effect():
    mcnemar = mcnemar_test([1, 1, 0, 0], [1, 1, 0, 1], [0, 1, 1, 0])
    assert mcnemar["a_correct_b_wrong"] == 2
    assert mcnemar["b_correct_a_wrong"] == 1
    assert 0 <= mcnemar["p_value"] <= 1
    wilcoxon = wilcoxon_signed_rank([0.8, 0.7, 0.9], [0.6, 0.7, 0.5])
    assert wilcoxon["median_difference"] == pytest.approx(0.2)
    assert wilcoxon["rank_biserial"] == pytest.approx(1.0)


@pytest.mark.parametrize("call", [
    lambda: classification_metrics([], []),
    lambda: precision_at_k([1], 0),
    lambda: recall_at_k([1], 0),
    lambda: cohens_kappa([], []),
    lambda: bootstrap_ci([], n_resamples=1),
    lambda: mcnemar_test([1], [1, 0], [1]),
    lambda: wilcoxon_signed_rank([1], [1, 2]),
])
def test_metrics_reject_invalid_inputs(call):
    with pytest.raises(ValueError):
        call()
