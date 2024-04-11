import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import interp1d

def loess_ci(x, y, percentile, frac, n_bootstrap=100):
    n_points = y.shape[0]

    # allow for multiple confidence interval percentiles
    n_ci = len(percentile)

    # one column for loess curve, two for each confidence interval bound
    y_out = np.zeros((n_points, 1+2*n_ci))

    # fit loess curve
    y_out[:,0] = lowess(y, x, return_sorted=False, frac=frac)

    # bootstrap confidence interval
    bootstrap_estimates = np.zeros((n_points, n_bootstrap))
    if(n_bootstrap>0):
        for i_bs in range(n_bootstrap):
            idx_bs = np.random.choice(np.arange(n_points), size=n_points, replace=True)

            y_smoothed = lowess(y[idx_bs], x[idx_bs], return_sorted=False, frac=frac)

            tmp = interp1d(x[idx_bs], y_smoothed, fill_value='extrapolate')(x)
            bootstrap_estimates[:,i_bs] = tmp

        # calculate percentile bounds from bootstrap estimates
        for i_per, per in enumerate(percentile):
            y_out[:,(1+2*i_per):(3+2*i_per)] = np.nanpercentile(bootstrap_estimates, q=[50-per/2,50+per/2], axis=1).T

    return y_out