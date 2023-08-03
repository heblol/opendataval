from itertools import accumulate
from typing import Sequence

import numpy as np


def oned_twonn_clustering(inputs: Sequence[float]) -> tuple[list[int], list[int]]:
    """O(nlog(n)) sort, O(n) pass exact 2-NN clustering of 1 dimensional input data.

    References
    ----------
    .. [1] A. Grønlund, K. G. Larsen, A. Mathiasen, J. S. Nielsen, S. Schneider,
        and M. Song,
        Fast Exact k-Means, k-Medians and Bregman Divergence Clustering in 1D,
        arXiv.org, 2017. https://arxiv.org/abs/1701.07204.

    """
    sid = np.argsort(inputs, kind="stable")
    n = len(inputs)

    psums = list(accumulate((inputs[sid[i]] for i in range(n)), initial=0.0))
    psqsums = list(accumulate((inputs[sid[i]] ** 2 for i in range(n)), initial=0.0))

    def cost(i: int, j: int):
        sij = psums[j + 1] - psums[i]
        uij = sij / (j - i + 1)
        return (uij**2) * (j - i + 1) + (psqsums[j + 1] - psqsums[i]) - 2 * uij * sij

    split = min((i for i in range(1, n)), key=lambda i: cost(0, i - 1) + cost(i, n - 1))
    return sid[range(0, split)], sid[range(split, n)]


def f1_score(predicted: Sequence[float], actual: Sequence[float], total: int) -> float:
    """Computes the F1 score based on the indices of values found."""
    predicted_set, actual_set = set(predicted), set(actual)

    tp, fp, fn = 0, 0, 0
    for i in range(total):
        if i in predicted_set and i in actual_set:
            tp += 1
        elif i in predicted_set:
            fp += 1
        elif i in actual_set:
            fn += 1
    return 2 * tp / (2 * tp + fp + fn)
