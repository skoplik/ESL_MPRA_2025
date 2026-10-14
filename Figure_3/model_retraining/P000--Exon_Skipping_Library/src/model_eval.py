import numpy as np
import scipy.stats as stats


def r2(x,y):
    """ computes Pearson r^2 value """
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    return r_value**2


def calc_r(x,y):
    """ computes Pearson r value """
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    return r_value


def mse(x, y):
    """ Computes mean squared error """
    return np.mean((x - y) ** 2)
