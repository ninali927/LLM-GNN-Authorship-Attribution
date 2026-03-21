import numpy as np
from WAN.markov_normalization import compute_stationary_distribution


def Hellinger_Distance(P1, P2):
    """
    H(P1, P2) = (1 / sqrt(2)) * sqrt( sum_i ( sqrt(pi1_i) - sqrt(pi2_i) )^2 )

    where pi1 and pi2 are the stationary distributions of P1 and P2.
    """
    pi1 = compute_stationary_distribution(P1)
    pi2 = compute_stationary_distribution(P2)

    value = 0.0

    n = len(pi1)

    for i in range(n):
        value += (np.sqrt(pi1[i]) - np.sqrt(pi2[i])) ** 2

    value = (1 / np.sqrt(2)) * np.sqrt(value)

    return value