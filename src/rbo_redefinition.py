import math
import numpy as np
import os
from pygam import LogisticGAM, s
import sys
from typing import Any, List, Set, Tuple
import warnings


class _HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def sigmoid(x) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def set_construction(collection, depth) -> Set[Any]:
    result = set()
    for i in range(depth):
        if i >= len(collection):
            break
        result.add(collection[i])
    return result


def overlap(list1, list2, depth) -> int:
    set1 = set_construction(list1, depth)
    set2 = set_construction(list2, depth)
    return len(set1.intersection(set2))


def real_rbo(
    list1, list2, p, shorter_prefix_depth=None
) -> Tuple[float, List[float], List[float]]:
    observed_agreements_to_s = []
    real_agreements_beyond_s = []
    maximal_depth = min(len(list1), len(list2))
    pairs_depth_agreement = [
        (d, (1.0 * overlap(list1, list2, d)) / d) for d in range(1, maximal_depth + 1)
    ]

    if shorter_prefix_depth is not None:
        observed_agreements_to_s.extend(
            list(
                map(lambda pair: pair[1], pairs_depth_agreement[:shorter_prefix_depth])
            )
        )
        real_agreements_beyond_s.extend(
            list(
                map(lambda pair: pair[1], pairs_depth_agreement[shorter_prefix_depth:])
            )
        )

    result = (1 - p) * sum(
        (p ** (pair[0] - 1)) * pair[1] for pair in pairs_depth_agreement
    )
    return result, observed_agreements_to_s, real_agreements_beyond_s


def rbo_min(S, L, p) -> float:
    k = len(S)
    overlap_at_k = overlap(S, L, k)

    first_term = sum(
        (overlap(S, L, d) - overlap_at_k) * ((p ** d) / d) for d in range(1, k + 1)
    )
    second_term = math.log(1 - p) * overlap_at_k

    return ((1 - p) / p) * (first_term - second_term)


def rbo_res(S, L, p) -> float:
    len_S = len(S)
    len_L = len(L)
    overlap_at_l = overlap(S, L, len_L)
    f = len_S + len_L - overlap_at_l

    first_term = len_S * sum((p ** d) / d for d in range(len_S + 1, f + 1))
    second_term = len_L * sum((p ** d) / d for d in range(len_L + 1, f + 1))
    third_term = overlap_at_l * (
        math.log(1 / (1 - p)) - sum((p ** d) / d for d in range(1, f + 1))
    )

    return (
        (p ** len_S)
        + (p ** len_L)
        - (p ** f)
        - ((1 - p) / p) * (first_term + second_term + third_term)
    )


def rbo_ext_original(S, L, p) -> Tuple[float, List[float], float]:
    len_S = len(S)
    len_L = len(L)
    observed_overlaps = []
    assumed_agreements_second_section = []

    first_term = 0.0
    for d in range(1, len_L + 1):
        current_overlap = overlap(S, L, d)
        observed_overlaps.append(current_overlap)
        first_term += (p ** d) * ((1.0 * current_overlap) / d)

    overlap_at_s = observed_overlaps[len_S - 1]
    overlap_at_l = observed_overlaps[len_L - 1]

    second_term = 0.0
    for d in range(len_S + 1, len_L + 1):
        extrapolation_term = (d - len_S) * ((1.0 * overlap_at_s) / len_S)
        second_term += (p ** d) * (extrapolation_term / d)

        current_assumed_agreement = (observed_overlaps[d - 1] + extrapolation_term) / d
        assumed_agreements_second_section.append(current_assumed_agreement)

    third_term = (overlap_at_s / len_S) + ((overlap_at_l - overlap_at_s) / len_L)
    if len(assumed_agreements_second_section) != 0:
        assert math.fabs(third_term - assumed_agreements_second_section[-1]) <= 0.000001

    result = ((1 - p) / p) * (first_term + second_term) + (p ** len_L) * third_term
    return result, assumed_agreements_second_section, third_term


def rbo_ext_previous_value(S, L, p, maximal_depth) -> Tuple[float, List[float]]:
    len_S = len(S)
    len_L = len(L)
    observed_overlaps = []
    observed_agreements = []

    for d in range(1, len_L + 1):
        current_overlap = overlap(S, L, d)
        observed_overlaps.append(current_overlap)
        observed_agreements.append((1.0 * current_overlap) / d)

    assumed_agreements = []
    estimated_membership_probs_second_section = []
    estimated_membership_probs_third_section = []
    for d in range(len_S + 1, maximal_depth + 1):
        if d <= len_L:
            if len(estimated_membership_probs_second_section) == 0:
                estimated_membership_probs_second_section.append(
                    observed_agreements[len_S - 1]
                )
            assumed_overlap = observed_overlaps[d - 1] + sum(
                estimated_membership_probs_second_section
            )
            assumed_agreement = assumed_overlap / d
            estimated_membership_probs_second_section.append(assumed_agreement)
        else:
            if len(estimated_membership_probs_third_section) == 0:
                if len(estimated_membership_probs_second_section) == 0:
                    estimated_membership_probs_third_section.append(
                        observed_agreements[len_S - 1]
                    )
                else:
                    estimated_membership_probs_third_section.append(
                        estimated_membership_probs_second_section[-1]
                    )
                    estimated_membership_probs_second_section.pop()
            assumed_overlap = (
                observed_overlaps[-1]
                + sum(estimated_membership_probs_second_section)
                + sum(
                    list(map(lambda x: x * x, estimated_membership_probs_third_section))
                )
            )
            assumed_agreement = assumed_overlap / d
            estimated_membership_probs_third_section.append(assumed_agreement)
        assumed_agreements.append(assumed_agreement)

    first_term = sum((p ** d) * observed_agreements[d - 1] for d in range(1, len_S + 1))
    second_term = sum(
        (p ** pair[0]) * pair[1]
        for pair in list(zip(range(len_S + 1, maximal_depth + 1), assumed_agreements))
    )

    result = ((1 - p) / p) * (first_term + second_term)
    return result, assumed_agreements


def rbo_ext_logit(S, L, p, maximal_depth) -> Tuple[float, List[float], List[float]]:
    len_S = len(S)
    len_L = len(L)
    observed_overlaps = []
    observed_agreements = []

    for d in range(1, len_L + 1):
        current_overlap = overlap(S, L, d)
        observed_overlaps.append(current_overlap)
        observed_agreements.append((1.0 * current_overlap) / d)

    coefficient, intercept = np.polyfit(
        range(1, len_S + 1), observed_agreements[:len_S], deg=1
    )
    fitted_agreements_seen_section = []
    for d in range(1, len_S + 1):
        fitted_agreements_seen_section.append(sigmoid(coefficient * d + intercept))

    assumed_agreements = []
    estimated_membership_probs_second_section = []
    estimated_membership_probs_third_section = []
    for d in range(len_S + 1, maximal_depth + 1):
        current_membership_prob = sigmoid(coefficient * d + intercept)
        if d <= len_L:
            estimated_membership_probs_second_section.append(current_membership_prob)
            assumed_overlap = observed_overlaps[d - 1] + sum(
                estimated_membership_probs_second_section
            )
        else:
            estimated_membership_probs_third_section.append(current_membership_prob)
            assumed_overlap = (
                observed_overlaps[-1]
                + sum(estimated_membership_probs_second_section)
                + sum(
                    list(map(lambda x: x * x, estimated_membership_probs_third_section))
                )
            )
        assumed_agreements.append(assumed_overlap / d)

    first_term = sum((p ** d) * observed_agreements[d - 1] for d in range(1, len_S + 1))
    second_term = sum(
        (p ** pair[0]) * pair[1]
        for pair in list(zip(range(len_S + 1, maximal_depth + 1), assumed_agreements))
    )

    result = ((1 - p) / p) * (first_term + second_term)
    return result, fitted_agreements_seen_section, assumed_agreements


def rbo_ext_gam(S, L, p, maximal_depth) -> Tuple[float, List[float], List[float]]:
    len_S = len(S)
    len_L = len(L)
    observed_overlaps = []
    observed_agreements = []

    for d in range(1, len_L + 1):
        current_overlap = overlap(S, L, d)
        observed_overlaps.append(current_overlap)
        observed_agreements.append((1.0 * current_overlap) / d)

    model = LogisticGAM(s(0, lam=0.01), fit_intercept=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with _HiddenPrints():
            model.fit(range(1, len_S + 1), observed_agreements[:len_S])

    fitted_agreements_seen_section = []
    for d in range(1, len_S + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with _HiddenPrints():
                fitted_agreements_seen_section.append(
                    sigmoid(model._linear_predictor(d)[0])
                )

    assumed_agreements = []
    estimated_membership_probs_second_section = []
    estimated_membership_probs_third_section = []
    for d in range(len_S + 1, maximal_depth + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with _HiddenPrints():
                current_membership_prob = sigmoid(model._linear_predictor(d)[0])
        if d <= len_L:
            estimated_membership_probs_second_section.append(current_membership_prob)
            assumed_overlap = observed_overlaps[d - 1] + sum(
                estimated_membership_probs_second_section
            )
        else:
            estimated_membership_probs_third_section.append(current_membership_prob)
            assumed_overlap = (
                observed_overlaps[-1]
                + sum(estimated_membership_probs_second_section)
                + sum(
                    list(map(lambda x: x * x, estimated_membership_probs_third_section))
                )
            )
        assumed_agreements.append(assumed_overlap / d)

    first_term = sum((p ** d) * observed_agreements[d - 1] for d in range(1, len_S + 1))
    second_term = sum(
        (p ** pair[0]) * pair[1]
        for pair in list(zip(range(len_S + 1, maximal_depth + 1), assumed_agreements))
    )

    result = ((1 - p) / p) * (first_term + second_term)
    return result, fitted_agreements_seen_section, assumed_agreements
