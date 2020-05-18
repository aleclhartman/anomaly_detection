import numpy as np
import pandas as pd

from env import host, user, password

def get_lower_and_upper_bounds(s, k=1.5):
    """
    Given a series and a multiplier, k, returns the upper and lower bounds for the series.
    """

    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    upper_bound = q3 + (k * iqr)
    lower_bound = q1 - (k * iqr)

    return lower_bound, upper_bound

def get_upper_outliers(s, k=1.5):
    """
    Given a series and a multiplier, k, returns the upper outliers for the
    series.

    The values returned will be either 0 (if the point is not an outlier), or a
    number that indicates how far away from the upper bound the observation is.
    """

    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    upper_bound = q3 + (k * iqr)
    return s.apply(lambda x: max([x - upper_bound, 0]))

def get_lower_outliers(s, k=1.5):
    """
    Given a series and a multiplier, k, returns the lower outliers for the
    series.

    The values returned will be either 0 (if the point is not an outlier), or a
    number that indicates how far away from the lower bound the observation is.
    """
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    lower_bound = q1 - (k * iqr)
    return s.apply(lambda x: min([x - lower_bound, 0]))