import numpy as np
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index

def CIndex_lifeline(hazards, labels, survtime_all):
    """
    Computes the Harrell's C-index using lifelines library.

    Args:
    - hazards (array-like): Hazard scores.
    - labels (array-like): Binary labels.
    - survtime_all (array-like): Survival times.

    Returns:
    - float: Harrell's C-index.
    """
    return(concordance_index(survtime_all, -hazards, labels))

def cox_log_rank(hazardsdata, labels, survtime_all):
    """
    Computes the p-value using the log-rank test on Cox hazards.

    Args:
    - hazardsdata (array-like): Hazard scores.
    - labels (array-like): Binary labels.
    - survtime_all (array-like): Survival times.

    Returns:
    - float: p-value from the log-rank test.
    """

    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    idx = hazards_dichotomize == 0
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return(pvalue_pred)

def accuracy_cox(hazardsdata, labels):
    """
    Computes the accuracy of Cox hazards-based survival events estimation against true events.

    Args:
    - hazardsdata (array-like): Hazard scores.
    - labels (array-like): Binary labels.

    Returns:
    - float: Accuracy of the Cox hazards-based survival events estimation.
    """
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    correct = np.sum(hazards_dichotomize == labels)
    return correct / len(labels)