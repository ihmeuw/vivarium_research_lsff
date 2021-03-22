import numpy as np, pandas as pd
import scipy.stats

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html

def beta_a_b_from_mean_var(mean, variance):
    """
    Returns the parameters a=alpha and b=beta for a beta distribution
    with the specified mean and variance.
    """
    if mean <= 0 or mean >= 1:
        raise ValueError("Mean must be in the interval (0,1)")
    if variance >= mean*(1-mean):
        raise ValueError("Variance too large")

    # For derivations of these formulas, see:
    # https://en.wikipedia.org/wiki/Beta_distribution#Mean_and_variance
    a = mean*(mean*(1-mean)/variance - 1)
    b = (1-mean)*(mean*(1-mean)/variance - 1)
    return a, b

def normal_stdev_from_mean_quantile(mean, quantile, quantile_rank):
    """
    Computes the standard deviation of a normal distribution that has the
    specified mean and quantile.
    """
    # If q = quantile, mu = mean, and sigma = std deviation, then
    # q = mu + q'*sigma, where q' is the standard normal quantile
    # and q is the transformed quantile, so sigma = (q-mu)/q'
    return (quantile - mean) / scipy.stats.norm().ppf(quantile_rank)

def beta_from_mean_approx_quantile(mean, approx_quantile, quantile_rank):
    """
    Returns a scipy.stats Beta distribution with the specified mean and a
    quantile of rank quantile_rank approximately equal to approx_quantile.
    This is achieved by specifying that the variance of the Beta distribution
    is equal to the variance of a normal distribution with the same mean and
    the specified quantile.
    
    Example usage - distribution of parmeter 'a' (eats_fortified) for India (Rajsathan):
    mean = 6.3 / 100 # mean = 6.3%
    q_975 = 7.9 / 100 # 97.5th percentile = 7.9%
    india_a_distribution = beta_from_mean_approx_quantile(mean, q_975, 0.975)
    """
    variance = normal_stdev_from_mean_quantile(mean, approx_quantile, quantile_rank)**2
    a,b = beta_a_b_from_mean_var(mean, variance)
    return scipy.stats.beta(a,b)


