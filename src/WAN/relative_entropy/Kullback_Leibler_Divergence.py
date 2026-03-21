import numpy as np
from WAN.markov_normalization import compute_stationary_distribution


def Kullback_Leibler_Divergence(P1, P2, epsilon=1e-12):
    """
    KL(P1 || P2) = sum_i pi1_i log(pi1_i / pi2_i)

    where pi1 and pi2 are the stationary distributions of P1 and P2.
    """
    pi1 = compute_stationary_distribution(P1)
    pi2 = compute_stationary_distribution(P2)

    value = 0.0
    n = len(pi1)

    for i in range(n):
        p = pi1[i]
        q = pi2[i]

        if p > 0:
            q_safe = max(q, epsilon)
            value += p * np.log(p / q_safe)

    return value