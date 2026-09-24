import numpy as np
from scipy.stats import chi2


def calculate_chi2_pvalue_const(y, uy, sys_error=0):
    """
    Calculate the chi-square value, degrees of freedom, and p-value for a constant fit.

    Parameters:
    y (array-like): Observed values.
    uy (array-like): Uncertainties of the observed values.
    sys_error (float, optional): Systematic error. Default is 0.

    Returns:
    tuple: A tuple containing the chi-square value, degrees of freedom, and p-value.
    """
    y, uy = np.array(y), np.array(uy)
    
    uncertainty = np.sqrt((sys_error * y)**2 + uy**2)
    
    mean_y     = (y/uncertainty**2).sum() / (1/uncertainty**2).sum()
    mean_y_err = np.sqrt(1/np.sum(1/uncertainty**2))
    
    chi2_value = np.sum((y - mean_y)**2/uncertainty**2)
    ndf = len(y) - 1
    pvalue = chi2.sf(x=chi2_value, df=ndf)
    return chi2_value, ndf, pvalue

def calculate_chi2_pvalue_fun(x, y, uy, f, params, sys_error=0):
    """
    Calculate the chi-square value, degrees of freedom, and p-value for a given set of data and model.

    Parameters:
    - x: array-like, x-values of the data
    - y: array-like, y-values of the data
    - uy: array-like, uncertainties of the y-values
    - f: function, model function that takes parameters and x-values as inputs
    - params: array-like, parameters for the model function
    - sys_error: float, systematic error (default: 0)

    Returns:
    - chi2_value: float, chi-square value
    - ndf: int, degrees of freedom
    - pvalue: float, p-value
    """
    x, y, uy = np.array(x), np.array(y), np.array(uy)
    
    uncertainty = np.sqrt((sys_error * y)**2 + uy**2)
    
    mean_y     = f(params, x)
    
    chi2_value = np.sum((y - mean_y)**2/uncertainty**2)
    ndf = len(y) - 1
    pvalue = chi2.sf(x=chi2_value, df=ndf)
    return chi2_value, ndf, pvalue

def weighted_average(y, uy, sys_error=0):
    """
    Calculate the weighted average of a set of values.

    Parameters:
    y (array-like): The values to be averaged.
    uy (array-like): The uncertainties associated with each value.
    sys_error (float, optional): The systematic error to be considered. Default is 0.

    Returns:
    float: The weighted average.
    float: The uncertainty of the weighted average.
    """
    y, uy = np.array(y), np.array(uy)
    
    uncertainty = np.sqrt((sys_error * y)**2 + uy**2)
    return (y/uncertainty**2).sum() / (1/uncertainty**2).sum(), np.sqrt(1/np.sum(1/uncertainty**2))

