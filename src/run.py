import json
import math
import matplotlib.pyplot as plt
import numpy as np
import random
from kyordanov.src.rbo_redefinition import (
    real_rbo,
    rbo_ext_original,
    rbo_ext_previous_value,
    rbo_ext_logit,
    rbo_ext_gam,
)


def config_run(file_path, p, l_ceiling, s_medium, s_large):
    assert 0.0 < p < 1.0
    file = open(file_path, "r")
    first_ranking = []
    second_ranking = []
    flag_append_first = True
    count_overall, count_small_s, count_medium_s, count_large_s = 0, 0, 0, 0

    (
        aggr_avg_overall_agreement_distance_original,
        aggr_avg_overall_agreement_distance_previous_value,
        aggr_avg_overall_agreement_distance_logit,
        aggr_avg_overall_agreement_distance_gam,
    ) = (0.0, 0.0, 0.0, 0.0)
    (
        aggr_avg_small_s_agreement_distance_original,
        aggr_avg_small_s_agreement_distance_previous_value,
        aggr_avg_small_s_agreement_distance_logit,
        aggr_avg_small_s_agreement_distance_gam,
    ) = (0.0, 0.0, 0.0, 0.0)
    (
        aggr_avg_medium_s_agreement_distance_original,
        aggr_avg_medium_s_agreement_distance_previous_value,
        aggr_avg_medium_s_agreement_distance_logit,
        aggr_avg_medium_s_agreement_distance_gam,
    ) = (0.0, 0.0, 0.0, 0.0)
    (
        aggr_avg_large_s_agreement_distance_original,
        aggr_avg_large_s_agreement_distance_previous_value,
        aggr_avg_large_s_agreement_distance_logit,
        aggr_avg_large_s_agreement_distance_gam,
    ) = (0.0, 0.0, 0.0, 0.0)

    (
        aggr_avg_overall_ext_distance_original,
        aggr_avg_overall_ext_distance_previous_value,
        aggr_avg_overall_ext_distance_logit,
        aggr_avg_overall_ext_distance_gam,
    ) = (0.0, 0.0, 0.0, 0.0)
    (
        aggr_avg_small_s_ext_distance_original,
        aggr_avg_small_s_ext_distance_previous_value,
        aggr_avg_small_s_ext_distance_logit,
        aggr_avg_small_s_ext_distance_gam,
    ) = (0.0, 0.0, 0.0, 0.0)
    (
        aggr_avg_medium_s_ext_distance_original,
        aggr_avg_medium_s_ext_distance_previous_value,
        aggr_avg_medium_s_ext_distance_logit,
        aggr_avg_medium_s_ext_distance_gam,
    ) = (0.0, 0.0, 0.0, 0.0)
    (
        aggr_avg_large_s_ext_distance_original,
        aggr_avg_large_s_ext_distance_previous_value,
        aggr_avg_large_s_ext_distance_logit,
        aggr_avg_large_s_ext_distance_gam,
    ) = (0.0, 0.0, 0.0, 0.0)

    (
        min_small_s_agreement_distance_previous_value,
        min_small_s_agreement_distance_logit,
        min_small_s_agreement_distance_gam,
        min_small_s_ext_distance_previous_value,
        min_small_s_ext_distance_logit,
        min_small_s_ext_distance_gam,
    ) = (1.5, 1.5, 1.5, 1.5, 1.5, 1.5)
    (
        min_medium_s_agreement_distance_previous_value,
        min_medium_s_agreement_distance_logit,
        min_medium_s_agreement_distance_gam,
        min_medium_s_ext_distance_previous_value,
        min_medium_s_ext_distance_logit,
        min_medium_s_ext_distance_gam,
    ) = (1.5, 1.5, 1.5, 1.5, 1.5, 1.5)
    (
        min_large_s_agreement_distance_previous_value,
        min_large_s_agreement_distance_logit,
        min_large_s_agreement_distance_gam,
        min_large_s_ext_distance_previous_value,
        min_large_s_ext_distance_logit,
        min_large_s_ext_distance_gam,
    ) = (1.5, 1.5, 1.5, 1.5, 1.5, 1.5)

    (
        max_small_s_agreement_distance_previous_value,
        max_small_s_agreement_distance_logit,
        max_small_s_agreement_distance_gam,
        max_small_s_ext_distance_previous_value,
        max_small_s_ext_distance_logit,
        max_small_s_ext_distance_gam,
    ) = (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
    (
        max_medium_s_agreement_distance_previous_value,
        max_medium_s_agreement_distance_logit,
        max_medium_s_agreement_distance_gam,
        max_medium_s_ext_distance_previous_value,
        max_medium_s_ext_distance_logit,
        max_medium_s_ext_distance_gam,
    ) = (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
    (
        max_large_s_agreement_distance_previous_value,
        max_large_s_agreement_distance_logit,
        max_large_s_agreement_distance_gam,
        max_large_s_ext_distance_previous_value,
        max_large_s_ext_distance_logit,
        max_large_s_ext_distance_gam,
    ) = (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0)

    for line in file:
        if line == "\n" and flag_append_first:
            flag_append_first = False
            continue
        if line == "\n":
            flag_append_first = True
            count_overall += 1
            value_s, results = single_pair_results(
                first_ranking, second_ranking, p, l_ceiling
            )

            aggr_avg_overall_agreement_distance_original += results["avg_agreement_distance_original"]
            aggr_avg_overall_agreement_distance_previous_value += results["avg_agreement_distance_previous_value"]
            aggr_avg_overall_agreement_distance_logit += results["avg_agreement_distance_logit"]
            aggr_avg_overall_agreement_distance_gam += results["avg_agreement_distance_gam"]
            aggr_avg_overall_ext_distance_original += results["ext_distance_original"]
            aggr_avg_overall_ext_distance_previous_value += results["ext_distance_previous_value"]
            aggr_avg_overall_ext_distance_logit += results["ext_distance_logit"]
            aggr_avg_overall_ext_distance_gam += results["ext_distance_gam"]

            if value_s <= s_medium:
                count_small_s += 1

                aggr_avg_small_s_agreement_distance_original += results["avg_agreement_distance_original"]
                aggr_avg_small_s_agreement_distance_previous_value += results["avg_agreement_distance_previous_value"]
                aggr_avg_small_s_agreement_distance_logit += results["avg_agreement_distance_logit"]
                aggr_avg_small_s_agreement_distance_gam += results["avg_agreement_distance_gam"]
                aggr_avg_small_s_ext_distance_original += results["ext_distance_original"]
                aggr_avg_small_s_ext_distance_previous_value += results["ext_distance_previous_value"]
                aggr_avg_small_s_ext_distance_logit += results["ext_distance_logit"]
                aggr_avg_small_s_ext_distance_gam += results["ext_distance_gam"]

                if results["avg_agreement_distance_previous_value"] <= min_small_s_agreement_distance_previous_value:
                    best_performing_agreement_small_s_previous_value = results["info"]
                    min_small_s_agreement_distance_previous_value = results["avg_agreement_distance_previous_value"]
                if results["avg_agreement_distance_previous_value"] >= max_small_s_agreement_distance_previous_value:
                    worst_performing_agreement_small_s_previous_value = results["info"]
                    max_small_s_agreement_distance_previous_value = results["avg_agreement_distance_previous_value"]

                if results["avg_agreement_distance_logit"] <= min_small_s_agreement_distance_logit:
                    best_performing_agreement_small_s_logit = results["info"]
                    min_small_s_agreement_distance_logit = results["avg_agreement_distance_logit"]
                if results["avg_agreement_distance_logit"] >= max_small_s_agreement_distance_logit:
                    worst_performing_agreement_small_s_logit = results["info"]
                    max_small_s_agreement_distance_logit = results["avg_agreement_distance_logit"]

                if results["avg_agreement_distance_gam"] <= min_small_s_agreement_distance_gam:
                    best_performing_agreement_small_s_gam = results["info"]
                    min_small_s_agreement_distance_gam = results["avg_agreement_distance_gam"]
                if results["avg_agreement_distance_gam"] >= max_small_s_agreement_distance_gam:
                    worst_performing_agreement_small_s_gam = results["info"]
                    max_small_s_agreement_distance_gam = results["avg_agreement_distance_gam"]

                if results["ext_distance_previous_value"] <= min_small_s_ext_distance_previous_value:
                    best_performing_ext_small_s_previous_value = results["info"]
                    min_small_s_ext_distance_previous_value = results["ext_distance_previous_value"]
                if results["ext_distance_previous_value"] >= max_small_s_ext_distance_previous_value:
                    worst_performing_ext_small_s_previous_value = results["info"]
                    max_small_s_ext_distance_previous_value = results["ext_distance_previous_value"]

                if results["ext_distance_logit"] <= min_small_s_ext_distance_logit:
                    best_performing_ext_small_s_logit = results["info"]
                    min_small_s_ext_distance_logit = results["ext_distance_logit"]
                if results["ext_distance_logit"] >= max_small_s_ext_distance_logit:
                    worst_performing_ext_small_s_logit = results["info"]
                    max_small_s_ext_distance_logit = results["ext_distance_logit"]

                if results["ext_distance_gam"] <= min_small_s_ext_distance_gam:
                    best_performing_ext_small_s_gam = results["info"]
                    min_small_s_ext_distance_gam = results["ext_distance_gam"]
                if results["ext_distance_gam"] >= max_small_s_ext_distance_gam:
                    worst_performing_ext_small_s_gam = results["info"]
                    max_small_s_ext_distance_gam = results["ext_distance_gam"]

            elif (s_medium + 1) <= value_s <= s_large:
                count_medium_s += 1

                aggr_avg_medium_s_agreement_distance_original += results["avg_agreement_distance_original"]
                aggr_avg_medium_s_agreement_distance_previous_value += results["avg_agreement_distance_previous_value"]
                aggr_avg_medium_s_agreement_distance_logit += results["avg_agreement_distance_logit"]
                aggr_avg_medium_s_agreement_distance_gam += results["avg_agreement_distance_gam"]
                aggr_avg_medium_s_ext_distance_original += results["ext_distance_original"]
                aggr_avg_medium_s_ext_distance_previous_value += results["ext_distance_previous_value"]
                aggr_avg_medium_s_ext_distance_logit += results["ext_distance_logit"]
                aggr_avg_medium_s_ext_distance_gam += results["ext_distance_gam"]

                if results["avg_agreement_distance_previous_value"] <= min_medium_s_agreement_distance_previous_value:
                    best_performing_agreement_medium_s_previous_value = results["info"]
                    min_medium_s_agreement_distance_previous_value = results["avg_agreement_distance_previous_value"]
                if results["avg_agreement_distance_previous_value"] >= max_medium_s_agreement_distance_previous_value:
                    worst_performing_agreement_medium_s_previous_value = results["info"]
                    max_medium_s_agreement_distance_previous_value = results["avg_agreement_distance_previous_value"]

                if results["avg_agreement_distance_logit"] <= min_medium_s_agreement_distance_logit:
                    best_performing_agreement_medium_s_logit = results["info"]
                    min_medium_s_agreement_distance_logit = results["avg_agreement_distance_logit"]
                if results["avg_agreement_distance_logit"] >= max_medium_s_agreement_distance_logit:
                    worst_performing_agreement_medium_s_logit = results["info"]
                    max_medium_s_agreement_distance_logit = results["avg_agreement_distance_logit"]

                if results["avg_agreement_distance_gam"] <= min_medium_s_agreement_distance_gam:
                    best_performing_agreement_medium_s_gam = results["info"]
                    min_medium_s_agreement_distance_gam = results["avg_agreement_distance_gam"]
                if results["avg_agreement_distance_gam"] >= max_medium_s_agreement_distance_gam:
                    worst_performing_agreement_medium_s_gam = results["info"]
                    max_medium_s_agreement_distance_gam = results["avg_agreement_distance_gam"]

                if results["ext_distance_previous_value"] <= min_medium_s_ext_distance_previous_value:
                    best_performing_ext_medium_s_previous_value = results["info"]
                    min_medium_s_ext_distance_previous_value = results["ext_distance_previous_value"]
                if results["ext_distance_previous_value"] >= max_medium_s_ext_distance_previous_value:
                    worst_performing_ext_medium_s_previous_value = results["info"]
                    max_medium_s_ext_distance_previous_value = results["ext_distance_previous_value"]

                if results["ext_distance_logit"] <= min_medium_s_ext_distance_logit:
                    best_performing_ext_medium_s_logit = results["info"]
                    min_medium_s_ext_distance_logit = results["ext_distance_logit"]
                if results["ext_distance_logit"] >= max_medium_s_ext_distance_logit:
                    worst_performing_ext_medium_s_logit = results["info"]
                    max_medium_s_ext_distance_logit = results["ext_distance_logit"]

                if results["ext_distance_gam"] <= min_medium_s_ext_distance_gam:
                    best_performing_ext_medium_s_gam = results["info"]
                    min_medium_s_ext_distance_gam = results["ext_distance_gam"]
                if results["ext_distance_gam"] >= max_medium_s_ext_distance_gam:
                    worst_performing_ext_medium_s_gam = results["info"]
                    max_medium_s_ext_distance_gam = results["ext_distance_gam"]

            else:
                count_large_s += 1

                aggr_avg_large_s_agreement_distance_original += results["avg_agreement_distance_original"]
                aggr_avg_large_s_agreement_distance_previous_value += results["avg_agreement_distance_previous_value"]
                aggr_avg_large_s_agreement_distance_logit += results["avg_agreement_distance_logit"]
                aggr_avg_large_s_agreement_distance_gam += results["avg_agreement_distance_gam"]
                aggr_avg_large_s_ext_distance_original += results["ext_distance_original"]
                aggr_avg_large_s_ext_distance_previous_value += results["ext_distance_previous_value"]
                aggr_avg_large_s_ext_distance_logit += results["ext_distance_logit"]
                aggr_avg_large_s_ext_distance_gam += results["ext_distance_gam"]

                if results["avg_agreement_distance_previous_value"] <= min_large_s_agreement_distance_previous_value:
                    best_performing_agreement_large_s_previous_value = results["info"]
                    min_large_s_agreement_distance_previous_value = results["avg_agreement_distance_previous_value"]
                if results["avg_agreement_distance_previous_value"] >= max_large_s_agreement_distance_previous_value:
                    worst_performing_agreement_large_s_previous_value = results["info"]
                    max_large_s_agreement_distance_previous_value = results["avg_agreement_distance_previous_value"]

                if results["avg_agreement_distance_logit"] <= min_large_s_agreement_distance_logit:
                    best_performing_agreement_large_s_logit = results["info"]
                    min_large_s_agreement_distance_logit = results["avg_agreement_distance_logit"]
                if results["avg_agreement_distance_logit"] >= max_large_s_agreement_distance_logit:
                    worst_performing_agreement_large_s_logit = results["info"]
                    max_large_s_agreement_distance_logit = results["avg_agreement_distance_logit"]

                if results["avg_agreement_distance_gam"] <= min_large_s_agreement_distance_gam:
                    best_performing_agreement_large_s_gam = results["info"]
                    min_large_s_agreement_distance_gam = results["avg_agreement_distance_gam"]
                if results["avg_agreement_distance_gam"] >= max_large_s_agreement_distance_gam:
                    worst_performing_agreement_large_s_gam = results["info"]
                    max_large_s_agreement_distance_gam = results["avg_agreement_distance_gam"]

                if results["ext_distance_previous_value"] <= min_large_s_ext_distance_previous_value:
                    best_performing_ext_large_s_previous_value = results["info"]
                    min_large_s_ext_distance_previous_value = results["ext_distance_previous_value"]
                if results["ext_distance_previous_value"] >= max_large_s_ext_distance_previous_value:
                    worst_performing_ext_large_s_previous_value = results["info"]
                    max_large_s_ext_distance_previous_value = results["ext_distance_previous_value"]

                if results["ext_distance_logit"] <= min_large_s_ext_distance_logit:
                    best_performing_ext_large_s_logit = results["info"]
                    min_large_s_ext_distance_logit = results["ext_distance_logit"]
                if results["ext_distance_logit"] >= max_large_s_ext_distance_logit:
                    worst_performing_ext_large_s_logit = results["info"]
                    max_large_s_ext_distance_logit = results["ext_distance_logit"]

                if results["ext_distance_gam"] <= min_large_s_ext_distance_gam:
                    best_performing_ext_large_s_gam = results["info"]
                    min_large_s_ext_distance_gam = results["ext_distance_gam"]
                if results["ext_distance_gam"] >= max_large_s_ext_distance_gam:
                    worst_performing_ext_large_s_gam = results["info"]
                    max_large_s_ext_distance_gam = results["ext_distance_gam"]

            first_ranking = []
            second_ranking = []
            continue
        elem = line.split("\n")[0]
        if flag_append_first:
            first_ranking.append(elem)
        else:
            second_ranking.append(elem)

    file.close()
    config_dict = {
        "p": p,
        "overall": {
            f"avg_of_{count_overall}_agreement_distance_original": aggr_avg_overall_agreement_distance_original / count_overall,
            f"avg_of_{count_overall}_agreement_distance_previous_value": aggr_avg_overall_agreement_distance_previous_value / count_overall,
            f"avg_of_{count_overall}_agreement_distance_logistic_regression": aggr_avg_overall_agreement_distance_logit / count_overall,
            f"avg_of_{count_overall}_agreement_distance_logistic_gam_regression": aggr_avg_overall_agreement_distance_gam / count_overall,
            f"avg_of_{count_overall}_ext_distance_original": aggr_avg_overall_ext_distance_original / count_overall,
            f"avg_of_{count_overall}_ext_distance_previous_value": aggr_avg_overall_ext_distance_previous_value / count_overall,
            f"avg_of_{count_overall}_ext_distance_logistic_regression": aggr_avg_overall_ext_distance_logit / count_overall,
            f"avg_of_{count_overall}_ext_distance_logistic_gam_regression": aggr_avg_overall_ext_distance_gam / count_overall,
        },
        f"small_s_to_{s_medium}": {
            f"avg_of_{count_small_s}_agreement_distance_original": aggr_avg_small_s_agreement_distance_original / count_small_s,
            f"avg_of_{count_small_s}_agreement_distance_previous_value": aggr_avg_small_s_agreement_distance_previous_value / count_small_s,
            f"avg_of_{count_small_s}_agreement_distance_logistic_regression": aggr_avg_small_s_agreement_distance_logit / count_small_s,
            f"avg_of_{count_small_s}_agreement_distance_logistic_gam_regression": aggr_avg_small_s_agreement_distance_gam / count_small_s,
            f"avg_of_{count_small_s}_ext_distance_original": aggr_avg_small_s_ext_distance_original / count_small_s,
            f"avg_of_{count_small_s}_ext_distance_previous_value": aggr_avg_small_s_ext_distance_previous_value / count_small_s,
            f"avg_of_{count_small_s}_ext_distance_logistic_regression": aggr_avg_small_s_ext_distance_logit / count_small_s,
            f"avg_of_{count_small_s}_ext_distance_logistic_gam_regression": aggr_avg_small_s_ext_distance_gam / count_small_s,
            "worst_performing_agreement_previous_value": worst_performing_agreement_small_s_previous_value,
            "worst_performing_agreement_logistic_regression": worst_performing_agreement_small_s_logit,
            "worst_performing_agreement_logistic_gam_regression": worst_performing_agreement_small_s_gam,
            "worst_performing_ext_previous_value": worst_performing_ext_small_s_previous_value,
            "worst_performing_ext_logistic_regression": worst_performing_ext_small_s_logit,
            "worst_performing_ext_logistic_gam_regression": worst_performing_ext_small_s_gam,
            "best_performing_agreement_previous_value": best_performing_agreement_small_s_previous_value,
            "best_performing_agreement_logistic_regression": best_performing_agreement_small_s_logit,
            "best_performing_agreement_logistic_gam_regression": best_performing_agreement_small_s_gam,
            "best_performing_ext_previous_value": best_performing_ext_small_s_previous_value,
            "best_performing_ext_logistic_regression": best_performing_ext_small_s_logit,
            "best_performing_ext_logistic_gam_regression": best_performing_ext_small_s_gam,
        },
        f"medium_s_from_{s_medium + 1}_to_{s_large}": {
            f"avg_of_{count_medium_s}_agreement_distance_original": aggr_avg_medium_s_agreement_distance_original / count_medium_s,
            f"avg_of_{count_medium_s}_agreement_distance_previous_value": aggr_avg_medium_s_agreement_distance_previous_value / count_medium_s,
            f"avg_of_{count_medium_s}_agreement_distance_logistic_regression": aggr_avg_medium_s_agreement_distance_logit / count_medium_s,
            f"avg_of_{count_medium_s}_agreement_distance_logistic_gam_regression": aggr_avg_medium_s_agreement_distance_gam / count_medium_s,
            f"avg_of_{count_medium_s}_ext_distance_original": aggr_avg_medium_s_ext_distance_original / count_medium_s,
            f"avg_of_{count_medium_s}_ext_distance_previous_value": aggr_avg_medium_s_ext_distance_previous_value / count_medium_s,
            f"avg_of_{count_medium_s}_ext_distance_logistic_regression": aggr_avg_medium_s_ext_distance_logit / count_medium_s,
            f"avg_of_{count_medium_s}_ext_distance_logistic_gam_regression": aggr_avg_medium_s_ext_distance_gam / count_medium_s,
            "worst_performing_agreement_previous_value": worst_performing_agreement_medium_s_previous_value,
            "worst_performing_agreement_logistic_regression": worst_performing_agreement_medium_s_logit,
            "worst_performing_agreement_logistic_gam_regression": worst_performing_agreement_medium_s_gam,
            "worst_performing_ext_previous_value": worst_performing_ext_medium_s_previous_value,
            "worst_performing_ext_logistic_regression": worst_performing_ext_medium_s_logit,
            "worst_performing_ext_logistic_gam_regression": worst_performing_ext_medium_s_gam,
            "best_performing_agreement_previous_value": best_performing_agreement_medium_s_previous_value,
            "best_performing_agreement_logistic_regression": best_performing_agreement_medium_s_logit,
            "best_performing_agreement_logistic_gam_regression": best_performing_agreement_medium_s_gam,
            "best_performing_ext_previous_value": best_performing_ext_medium_s_previous_value,
            "best_performing_ext_logistic_regression": best_performing_ext_medium_s_logit,
            "best_performing_ext_logistic_gam_regression": best_performing_ext_medium_s_gam,
        },
        f"large_s_beyond_{s_large + 1}": {
            f"avg_of_{count_large_s}_agreement_distance_original": aggr_avg_large_s_agreement_distance_original / count_large_s,
            f"avg_of_{count_large_s}_agreement_distance_previous_value": aggr_avg_large_s_agreement_distance_previous_value / count_large_s,
            f"avg_of_{count_large_s}_agreement_distance_logistic_regression": aggr_avg_large_s_agreement_distance_logit / count_large_s,
            f"avg_of_{count_large_s}_agreement_distance_logistic_gam_regression": aggr_avg_large_s_agreement_distance_gam / count_large_s,
            f"avg_of_{count_large_s}_ext_distance_original": aggr_avg_large_s_ext_distance_original / count_large_s,
            f"avg_of_{count_large_s}_ext_distance_previous_value": aggr_avg_large_s_ext_distance_previous_value / count_large_s,
            f"avg_of_{count_large_s}_ext_distance_logistic_regression": aggr_avg_large_s_ext_distance_logit / count_large_s,
            f"avg_of_{count_large_s}_ext_distance_logistic_gam_regression": aggr_avg_large_s_ext_distance_gam / count_large_s,
            "worst_performing_agreement_previous_value": worst_performing_agreement_large_s_previous_value,
            "worst_performing_agreement_logistic_regression": worst_performing_agreement_large_s_logit,
            "worst_performing_agreement_logistic_gam_regression": worst_performing_agreement_large_s_gam,
            "worst_performing_ext_previous_value": worst_performing_ext_large_s_previous_value,
            "worst_performing_ext_logistic_regression": worst_performing_ext_large_s_logit,
            "worst_performing_ext_logistic_gam_regression": worst_performing_ext_large_s_gam,
            "best_performing_agreement_previous_value": best_performing_agreement_large_s_previous_value,
            "best_performing_agreement_logistic_regression": best_performing_agreement_large_s_logit,
            "best_performing_agreement_logistic_gam_regression": best_performing_agreement_large_s_gam,
            "best_performing_ext_previous_value": best_performing_ext_large_s_previous_value,
            "best_performing_ext_logistic_regression": best_performing_ext_large_s_logit,
            "best_performing_ext_logistic_gam_regression": best_performing_ext_large_s_gam,
        },
    }
    return config_dict


def single_pair_results(ranking1, ranking2, p, l_ceiling):
    persistence_ranks = round(1 / (1 - p))
    infinity = min(len(ranking1), len(ranking2))

    len_L = random.randint(persistence_ranks, l_ceiling)
    len_S = random.randint(math.floor(0.75 * persistence_ranks), len_L)
    S = ranking1[:len_S]
    L = ranking2[:len_L]

    agreements_from_1_to_s, real_agreements_from_sP1_to_infinity = real_rbo(
        ranking1, ranking2, p, len(S)
    )[1:]
    real_rbo_score = rbo_ext_original(ranking1, ranking2, p)[0]
    original_ext, assumed_agreements_from_sP1_to_l, extrapolated_agreement_at_l = rbo_ext_original(S, L, p)
    previous_value_ext, assumed_agreements_from_sP1_to_infinity_pv = rbo_ext_previous_value(S, L, p, infinity)
    (
        logit_ext,
        fitted_agreements_from_1_to_s_logit,
        assumed_agreements_from_sP1_to_infinity_logit,
    ) = rbo_ext_logit(S, L, p, infinity)
    (
        gam_ext,
        fitted_agreements_from_1_to_s_gam,
        assumed_agreements_from_sP1_to_infinity_gam,
    ) = rbo_ext_gam(S, L, p, infinity)

    ext_distance_original = math.fabs(real_rbo_score - original_ext)
    ext_distance_previous_value = math.fabs(real_rbo_score - previous_value_ext)
    ext_distance_logit = math.fabs(real_rbo_score - logit_ext)
    ext_distance_gam = math.fabs(real_rbo_score - gam_ext)

    assumed_agreements_from_sP1_to_l.extend(
        extrapolated_agreement_at_l for _ in range(len_L + 1, infinity + 1)
    )

    avg_agreement_distance_original = np.mean(
        [
            math.fabs(x - y)
            for x, y in zip(
                real_agreements_from_sP1_to_infinity, assumed_agreements_from_sP1_to_l
            )
        ]
    )
    avg_agreement_distance_previous_value = np.mean(
        [
            math.fabs(x - y)
            for x, y in zip(
                real_agreements_from_sP1_to_infinity,
                assumed_agreements_from_sP1_to_infinity_pv,
            )
        ]
    )
    avg_agreement_distance_logit = np.mean(
        [
            math.fabs(x - y)
            for x, y in zip(
                real_agreements_from_sP1_to_infinity,
                assumed_agreements_from_sP1_to_infinity_logit,
            )
        ]
    )
    avg_agreement_distance_gam = np.mean(
        [
            math.fabs(x - y)
            for x, y in zip(
                real_agreements_from_sP1_to_infinity,
                assumed_agreements_from_sP1_to_infinity_gam,
            )
        ]
    )

    avg_fitted_agreement_distance_first_section_logit = np.mean(
        [
            math.fabs(x - y)
            for x, y in zip(agreements_from_1_to_s, fitted_agreements_from_1_to_s_logit)
        ]
    )
    avg_fitted_agreement_distance_first_section_gam = np.mean(
        [
            math.fabs(x - y)
            for x, y in zip(agreements_from_1_to_s, fitted_agreements_from_1_to_s_gam)
        ]
    )

    result_dict = {
        "ext_distance_original": ext_distance_original,
        "ext_distance_previous_value": ext_distance_previous_value,
        "ext_distance_logit": ext_distance_logit,
        "ext_distance_gam": ext_distance_gam,
        "avg_agreement_distance_original": avg_agreement_distance_original,
        "avg_agreement_distance_previous_value": avg_agreement_distance_previous_value,
        "avg_agreement_distance_logit": avg_agreement_distance_logit,
        "avg_agreement_distance_gam": avg_agreement_distance_gam,
        "info": {
            "full_ranking_1": ranking1,
            "full_ranking_2": ranking2,
            "maximal_depth": infinity,
            "S": S,
            "L": L,
            "length_S": len_S,
            "length_L": len_L,
            "real_rbo_score": real_rbo_score,
            "agreements_from_1_to_s": agreements_from_1_to_s,
            "real_agreements_from_s+1_to_infinity": real_agreements_from_sP1_to_infinity,
            "original_ext": {
                "ext_value": original_ext,
                "distance_to_real_score": ext_distance_original,
                "assumed_agreements_from_s+1_to_infinity": assumed_agreements_from_sP1_to_l,
                "avg_distance_to_real_agreements": avg_agreement_distance_original,
            },
            "previous_value_ext": {
                "ext_value": previous_value_ext,
                "distance_to_real_score": ext_distance_previous_value,
                "assumed_agreements_from_s+1_to_infinity": assumed_agreements_from_sP1_to_infinity_pv,
                "avg_distance_to_real_agreements": avg_agreement_distance_previous_value,
            },
            "logistic_regression_ext": {
                "ext_value": logit_ext,
                "distance_to_real_score": ext_distance_logit,
                "assumed_agreements_from_s+1_to_infinity": assumed_agreements_from_sP1_to_infinity_logit,
                "avg_distance_to_real_agreements": avg_agreement_distance_logit,
                "fitted_agreements_from_1_to_s": fitted_agreements_from_1_to_s_logit,
                "avg_distance_for_fitted_agreements": avg_fitted_agreement_distance_first_section_logit,
            },
            "logistic_gam_regression_ext": {
                "ext_value": gam_ext,
                "distance_to_real_score": ext_distance_gam,
                "assumed_agreements_from_s+1_to_infinity": assumed_agreements_from_sP1_to_infinity_gam,
                "avg_distance_to_real_agreements": avg_agreement_distance_gam,
                "fitted_agreements_from_1_to_s": fitted_agreements_from_1_to_s_gam,
                "avg_distance_for_fitted_agreements": avg_fitted_agreement_distance_first_section_gam,
            },
        },
    }
    return len_S, result_dict


def plot_agreements(config_dict, p_value_str, keys_s, keys_criteria, farthest_depth):
    p_value = config_dict["p"]
    for s_type in keys_s:
        for criterion in keys_criteria:
            infinity = config_dict[s_type][criterion]["maximal_depth"]
            len_S = config_dict[s_type][criterion]["length_S"]
            len_L = config_dict[s_type][criterion]["length_L"]

            agreements_to_s = config_dict[s_type][criterion]["agreements_from_1_to_s"]
            fitted_agreements_logit = config_dict[s_type][criterion]["logistic_regression_ext"]["fitted_agreements_from_1_to_s"]
            fitted_agreements_gam = config_dict[s_type][criterion]["logistic_gam_regression_ext"]["fitted_agreements_from_1_to_s"]

            real_agreements = config_dict[s_type][criterion]["real_agreements_from_s+1_to_infinity"]
            assumed_agreements_og = config_dict[s_type][criterion]["original_ext"]["assumed_agreements_from_s+1_to_infinity"]
            assumed_agreements_pv = config_dict[s_type][criterion]["previous_value_ext"]["assumed_agreements_from_s+1_to_infinity"]
            assumed_agreements_logit = config_dict[s_type][criterion]["logistic_regression_ext"]["assumed_agreements_from_s+1_to_infinity"]
            assumed_agreements_gam = config_dict[s_type][criterion]["logistic_gam_regression_ext"]["assumed_agreements_from_s+1_to_infinity"]

            capitalized = list(
                map(
                    lambda s: s.capitalize() if s != "gam" else "GAM",
                    criterion.split("_"),
                )
            )

            fig_1, ax_1 = plt.subplots()
            ax_1.set_title(
                f"{' '.join(capitalized[:3])} Predictions for {'-'.join(capitalized[3:])} EXT\n(s = {len_S}, l = {len_L}, p = {p_value})"
            )
            ax_1.set_xlabel("Depth")
            ax_1.set_ylabel("Agreement")

            ax_1.plot(
                range(1, len_S + 1),
                agreements_to_s,
                color="c",
                label=f"Observed Agreements from 1 to s",
            )
            ax_1.plot(
                range(len_S + 1, farthest_depth + 1),
                real_agreements[: (farthest_depth - len_S)],
                color="k",
                label=f"Real Agreements Beyond s + 1",
            )
            ax_1.plot(
                range(len_S + 1, farthest_depth + 1),
                assumed_agreements_og[: (farthest_depth - len_S)],
                color="r",
                label=f"Agreements Assumed by Original EXT Beyond s + 1",
            )
            ax_1.plot(
                range(len_S + 1, farthest_depth + 1),
                assumed_agreements_pv[: (farthest_depth - len_S)],
                color="g",
                label=f"Agreements Assumed by Previous-Value EXT Beyond s + 1",
            )
            ax_1.plot(
                range(len_S + 1, farthest_depth + 1),
                assumed_agreements_logit[: (farthest_depth - len_S)],
                color="b",
                label=f"Agreements Assumed by Logistic-Regression EXT Beyond s + 1",
            )
            ax_1.plot(
                range(len_S + 1, farthest_depth + 1),
                assumed_agreements_gam[: (farthest_depth - len_S)],
                color="m",
                label=f"Agreements Assumed by Logistic-GAM-Regression EXT Beyond s + 1",
            )

            bbox_1 = ax_1.get_position()
            ax_1.set_position(
                [
                    bbox_1.x0,
                    bbox_1.y0 + bbox_1.height * 0.36,
                    bbox_1.width,
                    bbox_1.height * 0.64,
                ]
            )
            ax_1.legend(
                loc="lower center",
                bbox_to_anchor=(0.5, 0),
                bbox_transform=fig_1.transFigure,
            )

            plt.savefig(
                f"../figures/p{p_value_str}Value_s{s_type.split('_')[0].capitalize()}_{''.join(capitalized)}.png"
            )
            plt.close()

            fig_2, ax_2 = plt.subplots()
            ax_2.set_title(
                f"Fitted and Real Agreements for Logit and GAM\n(s = {len_S}, p = {p_value})"
            )
            ax_2.set_xlabel("Depth")
            ax_2.set_ylabel("Agreement")

            ax_2.plot(
                range(1, len_S + 1),
                agreements_to_s,
                color="c",
                marker="o",
                markersize=2,
                label=f"Observed Agreements from 1 to s",
            )
            ax_2.plot(
                range(1, len_S + 1),
                fitted_agreements_logit,
                color="b",
                marker="o",
                markersize=2,
                label=f"Agreements Fitted by Logistic-Regression EXT from 1 to s",
            )
            ax_2.plot(
                range(1, len_S + 1),
                fitted_agreements_gam,
                color="m",
                marker="o",
                markersize=2,
                label=f"Agreements Fitted by Logistic-GAM-Regression EXT from 1 to s",
            )

            bbox_2 = ax_2.get_position()
            ax_2.set_position(
                [
                    bbox_2.x0,
                    bbox_2.y0 + bbox_2.height * 0.19,
                    bbox_2.width,
                    bbox_2.height * 0.81,
                ]
            )
            ax_2.legend(
                loc="lower center",
                bbox_to_anchor=(0.5, 0),
                bbox_transform=fig_2.transFigure,
            )

            plt.savefig(
                f"../figures/p{p_value_str}Value_s{s_type.split('_')[0].capitalize()}_{''.join(capitalized)}_FittedAgreements.png"
            )
            plt.close()

            fig_3, ax_3 = plt.subplots()
            ax_3.set_title(
                f"{' '.join(capitalized[:3])} Predictions for {'-'.join(capitalized[3:])} EXT\n(s = {len_S}, l = {len_L}, p = {p_value})"
            )
            ax_3.set_xlabel("Depth")
            ax_3.set_ylabel("Agreement")

            ax_3.plot(
                range(1, len_S + 1),
                agreements_to_s,
                color="c",
                label=f"Observed Agreements from 1 to s",
            )
            ax_3.plot(
                range(len_S + 1, infinity + 1),
                real_agreements,
                color="k",
                label=f"Real Agreements Beyond s + 1",
            )
            ax_3.plot(
                range(len_S + 1, infinity + 1),
                assumed_agreements_og,
                color="r",
                label=f"Agreements Assumed by Original EXT Beyond s + 1",
            )
            ax_3.plot(
                range(len_S + 1, infinity + 1),
                assumed_agreements_pv,
                color="g",
                label=f"Agreements Assumed by Previous-Value EXT Beyond s + 1",
            )
            ax_3.plot(
                range(len_S + 1, infinity + 1),
                assumed_agreements_logit,
                color="b",
                label=f"Agreements Assumed by Logistic-Regression EXT Beyond s + 1",
            )
            ax_3.plot(
                range(len_S + 1, infinity + 1),
                assumed_agreements_gam,
                color="m",
                label=f"Agreements Assumed by Logistic-GAM-Regression EXT Beyond s + 1",
            )

            bbox_3 = ax_3.get_position()
            ax_3.set_position(
                [
                    bbox_3.x0,
                    bbox_3.y0 + bbox_3.height * 0.36,
                    bbox_3.width,
                    bbox_3.height * 0.64,
                ]
            )
            ax_3.legend(
                loc="lower center",
                bbox_to_anchor=(0.5, 0),
                bbox_transform=fig_3.transFigure,
            )

            plt.savefig(
                f"../figures/p{p_value_str}Value_s{s_type.split('_')[0].capitalize()}_{''.join(capitalized)}_ExhaustiveAgreements.png"
            )
            plt.close()


if __name__ == "__main__":
    random.seed(42)
    p_values = [0.8, 0.9, 0.95]
    data_file = "../data/data_5000pairs_2000length.txt"
    l_upper_threshold = 100
    s_medium_threshold = 15
    s_large_threshold = 45
    plotting_depth = 140

    for idx in range(len(p_values)):
        output_dict = config_run(
            data_file,
            p_values[idx],
            l_upper_threshold,
            s_medium_threshold,
            s_large_threshold,
        )
        value_str = (
            f"{idx + 1}th"
            if idx >= 3
            else "3rd" if idx == 2 else "2nd" if idx == 1 else "1st"
        )

        with open(
            f"../results/p{value_str}Value.json", "w", encoding="utf-8"
        ) as output_file:
            json.dump(output_dict, output_file, indent=4)

        s_keys = list(output_dict.keys())[2:]
        perform_keys = [
            "worst_performing_agreement_previous_value",
            "worst_performing_agreement_logistic_regression",
            "worst_performing_agreement_logistic_gam_regression",
            "best_performing_agreement_previous_value",
            "best_performing_agreement_logistic_regression",
            "best_performing_agreement_logistic_gam_regression",
        ]
        plot_agreements(output_dict, value_str, s_keys, perform_keys, plotting_depth)
