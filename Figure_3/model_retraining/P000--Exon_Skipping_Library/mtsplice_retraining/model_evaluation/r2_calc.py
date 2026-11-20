import scipy.stats as stats
import numpy as np
import pandas as pd


def calc_r2_with_nan(x, y, model=None):
    """
    get r^2 for each cell type separately, ignoring NaN values
    
    """
    scores = []
    if model is not None:
        x = model.predict(x, verbose=0)
    for i in range(5):
        nans = np.where(np.isnan(y[:,i]) == False)
        if isinstance(x, list):
            x = x[0]
        xi = x[:,i][nans[0]]
        yi = y[:,i][nans[0]]
        r_value = stats.linregress(xi, yi).rvalue
        scores.append(round(r_value**2, 3))    
    return scores


def calc_r2_with_nan_tsplice(x, y, model=None):
    """
    get r^2 for each cell type separately, ignoring NaN values
    
    for tsplice, add the mean cell type delta logit for each cell to the
    model outputs to get the final predictions    

    """
    scores = []
    if model is not None:
        p = model.predict(x, verbose=0)
        if isinstance(p, list):
            p = p[0]
    else:
        p = x
    for i in range(5):
        nans = np.where(np.isnan(y[:,i]) == False)
        if isinstance(x, list):
            x = x[0]
        pi = p[:,i][nans[0]]
        yi = y[:,i][nans[0]]
        r_value = stats.linregress(pi, yi).rvalue
        scores.append(round(r_value**2, 3))    
    return scores


def calc_r2_with_nan_spliceai(x, y, model=None):
    """
    since we are using the MMSplice data, 
    there shouldnt be any NaNs. However,
    check for those just in case
    """
    if model is None:
        predictions = x
    else:
        predictions = model.predict(x)
    if len(predictions.shape) == 1:
        predictions = np.reshape(predictions, (predictions.shape[0], 1))
    if isinstance(y, list):
        y = np.array(y).reshape(len(y), 1)
    if len(y.shape) == 1:
        y = np.reshape(y, (y.shape[0], 1))
    y_non_nans = np.sum(np.isnan(y), axis=1) == 0
    predictions = predictions[y_non_nans]
    y = y[y_non_nans]
    scores = []
    for i in range(y.shape[1]):
        r_value = stats.linregress(predictions[:,i], y[:,i]).rvalue
        scores.append(r_value**2)
    return scores


def calc_r_with_nan(x, y, model=None, stat="pearson"):
    """
    since we are using the MMSplice data, 
    there shouldnt be any NaNs. However,
    check for those just in case
    """
    if model is None:
        predictions = x
    else:
        predictions = model.predict(x)
    if len(predictions.shape) == 1:
        predictions = np.reshape(predictions, (predictions.shape[0], 1))
    if isinstance(y, pd.core.series.Series):
        y = y.to_numpy()
    if isinstance(x, pd.core.series.Series):
        x = x.to_numpy()
    if isinstance(y, list):
        y = np.array(y).reshape(len(y), 1)
    if len(y.shape) == 1:
        y = np.reshape(y, (y.shape[0], 1))
    if isinstance(x, list):
        x = np.array(x).reshape(len(x), 1)
    if len(x.shape) == 1:
        x = np.reshape(x, (x.shape[0], 1))
    scores = []
    for i in range(y.shape[1]):
        finite = np.logical_and(
            np.isfinite(y[:,i]),
            np.isfinite(x[:,i])
        )
        x_r = x[:,i][finite]
        y_r = y[:,i][finite]
        if stat.lower() == "pearson":
            r_value = stats.linregress(x_r, y_r).rvalue
        elif stat.lower() == "spearman":
            r_value = stats.spearmanr(x_r, y_r).statistic
        else:
            raise ValueError(
                f"Expected stat to be 'pearson' "
                f"or 'spearman' but got '{stat}'"
            )
        scores.append(r_value)
    return np.stack(scores)
