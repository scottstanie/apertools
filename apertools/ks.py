import numba
import numpy as np
from math import gcd


@numba.njit
def ks_2samp(data1, data2):
    """
    Compute the Kolmogorov-Smirnov statistic on 2 samples.
    This is a two-sided test for the null hypothesis that 2 independent samples
    are drawn from the same continuous distribution.  The alternative hypothesis
    is 'two-sided'
    Parameters
    ----------
    data1, data2 : array_like, 1-Dimensional
        Two arrays of sample observations assumed to be drawn from a continuous
        distribution, sample sizes can be different.
    Returns
    -------
    statistic : float
        KS statistic.
    pvalue : float
        Two-tailed p-value.
    See Also
    --------
    kstest, ks_1samp, epps_singleton_2samp, anderson_ksamp
    Notes
    -----
    This tests whether 2 samples are drawn from the same distribution. Note
    that, like in the case of the one-sample KS test, the distribution is
    assumed to be continuous.
    In the one-sided test, the alternative is that the empirical
    cumulative distribution function F(x) of the data1 variable is "less"
    or "greater" than the empirical cumulative distribution function G(x)
    of the data2 variable, ``F(x)<=G(x)``, resp. ``F(x)>=G(x)``.
    If the KS statistic is small or the p-value is high, then we cannot
    reject the hypothesis that the distributions of the two samples
    are the same.
    If the mode is 'auto', the computation is exact if the sample sizes are
    less than 10000.  For larger sizes, the computation uses the
    Kolmogorov-Smirnov distributions to compute an approximate value.
    The 'two-sided' 'exact' computation computes the complementary probability
    and then subtracts from 1.  As such, the minimum probability it can return
    is about 1e-16.  While the algorithm itself is exact, numerical
    errors may accumulate for large sample sizes.   It is most suited to
    situations in which one of the sample sizes is only a few thousand.
    We generally follow Hodges' treatment of Drion/Gnedenko/Korolyuk [1]_.
    References
    ----------
    .. [1] Hodges, J.L. Jr.,  "The Significance Probability of the Smirnov
           Two-Sample Test," Arkiv fiur Matematik, 3, No. 43 (1958), 469-86.
    Examples
    --------
    >>> from scipy import stats
    >>> np.random.seed(12345678)  #fix random seed to get the same result
    >>> n1 = 200  # size of first sample
    >>> n2 = 300  # size of second sample
    For a different distribution, we can reject the null hypothesis since the
    pvalue is below 1%:
    >>> rvs1 = stats.norm.rvs(size=n1, loc=0., scale=1)
    >>> rvs2 = stats.norm.rvs(size=n2, loc=0.5, scale=1.5)
    >>> stats.ks_2samp(rvs1, rvs2)
    (0.20833333333333334, 5.129279597781977e-05)
    For a slightly different distribution, we cannot reject the null hypothesis
    at a 10% or lower alpha since the p-value at 0.144 is higher than 10%
    >>> rvs3 = stats.norm.rvs(size=n2, loc=0.01, scale=1.0)
    >>> stats.ks_2samp(rvs1, rvs3)
    (0.10333333333333333, 0.14691437867433876)
    For an identical distribution, we cannot reject the null hypothesis since
    the p-value is high, 41%:
    >>> rvs4 = stats.norm.rvs(size=n2, loc=0.0, scale=1.0)
    >>> stats.ks_2samp(rvs1, rvs4)
    (0.07999999999999996, 0.41126949729859719)
    """
    data1 = np.sort(data1)
    data2 = np.sort(data2)
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    if min(n1, n2) == 0:
        raise ValueError("Data passed to ks_2samp must not be empty")

    data_all = np.concatenate((data1, data2))
    # using searchsorted solves equal data problem
    cdf1 = np.searchsorted(data1, data_all, side="right") / n1
    cdf2 = np.searchsorted(data2, data_all, side="right") / n2
    cddiffs = cdf1 - cdf2
    # minS = np.clip(-np.min(cddiffs), 0, 1)  # Ensure sign of minS is not negative.
    # np.clip not yet implemented in numba 0.53, next version
    minS = -np.min(cddiffs)
    if minS < 0:
        minS = 0
    elif minS > 1:
        minS = 1
    maxS = np.max(cddiffs)
    d = max(minS, maxS)
    g = gcd(n1, n2)
    prob = -np.inf

    # n1g = n1 // g
    # n2g = n2 // g
    # # If lcm(n1, n2) is too big, switch from exact to asymp
    # if n1g >= np.iinfo(np.int_).max / n2g:
    #     mode = "asymp"
    #     raise ValueError(
    #         f"Exact ks_2samp calculation not possible with samples sizes "
    #         f"{n1} and {n2}."
    #     )

    d, prob = _attempt_exact_2kssamp(n1, n2, g, d)

    # prob = np.clip(prob, 0, 1)
    if prob > 1:
        prob = 1
    elif prob < 0:
        prob = 0
    # return KstestResult(d, prob)
    return (d, prob)


@numba.njit
def _compute_prob_outside_square(n, h):
    """
    Compute the proportion of paths that pass outside the two diagonal lines.
    Parameters
    ----------
    n : integer
        n > 0
    h : integer
        0 <= h <= n
    Returns
    -------
    p : float
        The proportion of paths that pass outside the lines x-y = +/-h.
    """
    # Compute Pr(D_{n,n} >= h/n)
    # Prob = 2 * ( binom(2n, n-h) - binom(2n, n-2a) + binom(2n, n-3a) - ... )  / binom(2n, n)
    # This formulation exhibits subtractive cancellation.
    # Instead divide each term by binom(2n, n), then factor common terms
    # and use a Horner-like algorithm
    # P = 2 * A0 * (1 - A1*(1 - A2*(1 - A3*(1 - A4*(...)))))

    P = 0.0
    k = int(np.floor(n / h))
    while k >= 0:
        p1 = 1.0
        # Each of the Ai terms has numerator and denominator with h simple terms.
        for j in range(h):
            p1 = (n - k * h - j) * p1 / (n + k * h + j + 1)
        P = p1 * (1.0 - P)
        k -= 1
    return 2 * P


@numba.njit
def _attempt_exact_2kssamp(n1, n2, g, d):
    """Attempts to compute the exact 2sample probability.
    n1, n2 are the sample sizes
    g is the gcd(n1, n2)
    d is the computed max difference in ECDFs
    Returns (success, d, probability)
    """
    lcm = (n1 // g) * n2
    h = int(np.round(d * lcm))
    d = h * 1.0 / lcm
    if h == 0:
        return d, 1.0
    prob = _compute_prob_outside_square(n1, h)
    return d, prob
