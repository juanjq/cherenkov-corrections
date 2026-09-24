import numpy as np
from scipy.optimize import minimize

def lineFreeF(params, x):
    """
    Calculate the y-values of a line given the slope and intercept.

    Parameters:
    params (tuple): A tuple containing the slope and intercept of the line.
    x (list or array-like): The x-values at which to calculate the y-values.

    Returns:
    numpy.ndarray: The calculated y-values of the line.
    """
    slope, intercept = params
    return slope * np.array(x) + intercept

def lineConstF(params, x):
    """
    Returns an array of the same length as x, where each element is equal to the intercept value.

    Parameters:
    - params: The intercept value.
    - x: The input array.

    Returns:
    - An array of the same length as x, where each element is equal to the intercept value.
    """
    intercept = params
    return np.repeat(intercept, len(x))

def lineZeroF(params, x):
    """
    Calculate the y-coordinate of a point on a line with zero y-intercept.

    Parameters:
    params (float): The slope of the line.
    x (float): The x-coordinate of the point.

    Returns:
    float: The y-coordinate of the point on the line.
    """
    slope = params
    return slope * np.array(x)

def chi2F(params, f, x, y, uy):
    """
    Calculate the chi-squared value for a given set of parameters.

    Parameters:
        params (array-like): The parameters to be used in the function `f`.
        f (callable): The function to be evaluated.
        x (array-like): The independent variable values.
        y (array-like): The observed dependent variable values.
        uy (array-like): The uncertainties in the observed dependent variable values.

    Returns:
        float: The chi-squared value.
    """
    y_pred = f(params, x)
    residuals = (y - y_pred) / uy
    return np.sum(residuals**2)
    
def LRTFreeConst(x, y, uy):
    """
    Calculates the likelihood ratio test (LRT) statistic for the comparison of a free model and a constrained model.

    Parameters:
    x (array-like): The x-coordinates of the data points.
    y (array-like): The y-coordinates of the data points.
    uy (array-like): The uncertainties of the y-coordinates.

    Returns:
    lrt (float): The LRT statistic.
    free_params (array-like): The parameter values of the free model that minimize the chi-square function.
    const_params (array-like): The parameter values of the constrained model that minimize the chi-square function.
    """
    init_free = [1, 1]
    init_const = [np.mean(y)]
    
    LS_free  = minimize(chi2F, init_free,  args=(lineFreeF,  x, y, uy))
    LS_const = minimize(chi2F, init_const, args=(lineConstF, x, y, uy))
    
    lrt = np.sqrt(LS_const.fun - LS_free.fun) #* (LS_free.fun - LS_const.fun) / np.abs(LS_free.fun - LS_const.fun)

    return lrt, LS_free.x, LS_const.x

def LRTFreeZero(x, y, uy):
    """
    Calculates the likelihood ratio test (LRT) statistic for the hypothesis of a free model versus a zero model.
    
    Parameters:
    - x: array-like, x-coordinates of the data points
    - y: array-like, y-coordinates of the data points
    - uy: array-like, uncertainties of the y-coordinates
    
    Returns:
    - lrt: float, the LRT statistic
    - LS_free.x: array-like, the parameter values that minimize the chi-square function for the free model
    - LS_zero.x: array-like, the parameter values that minimize the chi-square function for the zero model
    """
    init_free = [1, 1]
    init_zero = [0]
    
    LS_free = minimize(chi2F, init_free, args=(lineFreeF, x, y, uy))
    LS_zero = minimize(chi2F, init_zero, args=(lineZeroF, x, y, uy))
    
    lrt = np.sqrt(LS_zero.fun - LS_free.fun)

    return lrt, LS_free.x, LS_zero.x

def likelihood_plaw(params, x, y, reference_intensity):
    """
    Calculate the likelihood of a power-law model given the parameters and observed data.

    Parameters:
        params (list): List of parameters [A, k] for the power-law model.
        x (array-like): Array of x-values.
        y (array-like): Array of observed y-values.

    Returns:
        float: The likelihood value.

    """
    norm, pindex = params[0], params[1]
    y_pred = powerlaw(x, norm, pindex)
    return np.sum((y - y_pred)**2)

def powerlaw(x, norm, pindex):
    """
    Power-law function.

    Parameters:
    x (float): The input value.
    norm (float): The amplitude.
    pindex (float): The power-law exponent.

    Returns:
    float: The calculated value of the power-law function.
    """
    return norm * (x) ** pindex

    
def expfunc(x, a, b):
    """
    Exponential function.

    Parameters:
    x (float): The input value.
    a (float): The amplitude.
    b (float): The exponential index.

    Returns:
    float: The calculated value of the exponential function.
    """    
    return a * np.exp(b * x)


def calc_light_yield(p0_fit, p1_fit, sigma_p0_fit, sigma_p1_fit, p0_ref):
    """
    Calculate the light yield based on the reference point, AR, alphaR, A2, and alpha2.

    Parameters:
    refpoint (float): The reference point.
    AR (float): The AR value.
    alphaR (float): The alphaR value.
    A2 (float): The A2 value.
    alpha2 (float): The alpha2 value.

    Returns:
    float: The calculated light yield.
    float: The error in the light yield
    """

    power_factor = (- 1 / ( 1 + p1_fit))

    ly    = (p0_fit / p0_ref) ** power_factor
    
    dlydA = (p0_fit / p0_ref) ** power_factor * 1 / (1 + p1_fit)    * (-1) / p0_fit
    dlyda = (p0_fit / p0_ref) ** power_factor * 1 / (1 + p1_fit)**2 * np.log(p0_fit / p0_ref) 

    delta_ly = np.sqrt((dlydA) ** 2 * sigma_p0_fit ** 2 + (dlyda) ** 2 * sigma_p1_fit ** 2)
    
    return ly, delta_ly
    
def straight_line(x, intercept, slope):
    """
    Straight line function.

    Parameters:
    x (float): The input value.
    intercept (float): The intercept.
    slope (float): The slope.

    Returns:
    float: The calculated value of the straight line function.
    """
    return intercept + slope * x
    
def pol2(x, a, b, c):
    """
    Calculates the value of a second-degree polynomial.

    Parameters:
    x (float): The input value.
    a (float): The coefficient of the constant term.
    b (float): The coefficient of the linear term.
    c (float): The coefficient of the quadratic term.

    Returns:
    float: The value of the polynomial at the given input value.
    """
    return a + b * x + c * x * x

def parabola_3points(x1, x2, x3, y1, y2, y3, uy1, uy2, uy3):
    """
    Calculates the parameters of a parabola that passes trough 3 points, given only the errors in the y-axis.

    Parameters:
    x1, x2, x3 (3-float): X-coordinates of the three points.
    y1, y2, y3 (3-float): Y-coordinates of the three points.
    uy1, uy2, uy3 (3-float): Error in Y-axis.

    Returns:
    a, b, c (3-float): The parameters of the polynomial.
    ua, ub, uc (3-float): The error on the parameters of the polynomial.
    """
    c = y1/(x1*x1-x1*x2-x1*x3+x2*x3)+y2/(-x1*x2+x1*x3+x2*x2-x2*x3)+y3/(x1*x2-x1*x3-x2*x3+x3*x3)
    b = (
        - (x2*y1)/( x1*x1 - x1*x2 - x1*x3 + x2*x3) - (x3*y1)/( x1*x1 - x1*x2 - x1*x3 + x2*x3) 
        - (x1*y2)/(-x1*x2 + x1*x3 + x2*x2 - x2*x3) - (x3*y2)/(-x1*x2 + x1*x3 + x2*x2 - x2*x3) 
        - (x1*y3)/( x1*x2 - x1*x3 - x2*x3 + x3*x3) - (x2*y3)/( x1*x2 - x1*x3 - x2*x3 + x3*x3)
    )
    a = (x2*x3*y1)/(x1*x1-x1*x2-x1*x3+x2*x3)+(x1*x3*y2)/(-x1*x2+x1*x3+x2*x2-x2*x3)+(x1*x2*y3)/(x1*x2-x1*x3-x2*x3+x3*x3)

    
    dcdy1 =  1/((x2-x1)*(x3-x1))
    dcdy2 = -1/((x2-x1)*(x3-x2))
    dcdy3 =  1/(x3**2+(-x2-x1)*x3+x1*x2)
    
    dbdy1 = -(x3+x2)/((x2-x1)*(x3-x1))
    dbdy2 =  (x3+x1)/((x2-x1)*(x3-x2))
    dbdy3 = -(x2+x1)/(x3**2+(-x2-x1)*x3+x1*x2)

    dady1 =  (x2*x3)/((x2-x1)*(x3-x1))
    dady2 = -(x1*x3)/((x2-x1)*(x3-x2))
    dady3 =  (x1*x2)/(x3**2+(-x2-x1)*x3+x1*x2)
    
    uc = np.sqrt(dcdy1**2 * uy1**2 + dcdy2**2 * uy2**2 + dcdy3**2 * uy3**2)
    ub = np.sqrt(dbdy1**2 * uy1**2 + dbdy2**2 * uy2**2 + dbdy3**2 * uy3**2)
    ua = np.sqrt(dady1**2 * uy1**2 + dady2**2 * uy2**2 + dady3**2 * uy3**2)
    
    return a, b, c, ua, ub, uc

# CHANGE AND ADD LIKELIHOOD METHOD
def get_roots_pol2(xref, yref, x1, x2, x3, y1, y2, y3, uy1, uy2, uy3):
    """
    Calculates the roots of a 2 degree polynomial that passes trough 3 points.

    Parameters:
    xref (float): A center or reference value in x-axis to find the nearest root around.
    yref (float): A baseline. Setting to 0 will calculate the usual roots, otherwise it 
    can calculate the roots at a certain y-value, that will be yref.
    x1, x2, x3 (3-float): X-coordinates of the three points.
    y1, y2, y3 (3-float): Y-coordinates of the three points.
    uy1, uy2, uy3 (3-float): Error in Y-axis.

    Returns:
    x0 (float): The root.
    delta_x0 (float): Estimated error on the root.
    """    
    yM1, yM2, yM3 = y1 + uy1, y2 + uy2, y3 + uy3 
    ym1, ym2, ym3 = y1 - uy1, y2 - uy2, y3 - uy3

    a, b, c, ua, ub, uc = parabola_3points(x1, x2, x3, y1, y2, y3, uy1, uy2, uy3)
    aM, bM, cM, _, _, _ = parabola_3points(x1, x2, x3, yM1, yM2, yM3, uy1, uy2, uy3)
    am, bm, cm, _, _, _ = parabola_3points(x1, x2, x3, ym1, ym2, ym3, uy1, uy2, uy3)
    a  = a  - yref
    aM = aM - yref
    am = am - yref

    sqr  = (b**2 - 4*a*c)
    sqrM = (bM**2 - 4*aM*cM)
    sqrm = (bm**2 - 4*am*cm)

    no_result   = False if sqr  >= 0 else True
    no_result_M = False if sqrM >= 0 else True
    no_result_m = False if sqrm >= 0 else True
    if not no_result:
        x1  = (-b + sqr**0.5) / (2 * c)
        x2  = (-b - sqr**0.5) / (2 * c)
        
        # dxda = 1/np.sqrt(sqr)
        # dxdb = b/sqr**(3/2)
        # dxdc = (6*a*b)/sqr**(5/2)
        # ux = np.sqrt(dxda**2 * ua**2 + dxdb**2 * ub**2 + dx1dc**2 * uc**2)
        
        roots_dist = [abs(x - xref) for x in [x1, x2]]
        roots_i = roots_dist.index(min(roots_dist))
        x0 = [x1, x2][roots_i]

    x0M = np.nan
    if (not no_result) and (not no_result_M):    
        xM1 = (-bM + sqrM**0.5) / (2 * cM)
        xM2 = (-bM - sqrM**0.5) / (2 * cM)

        roots_dist = [abs(x - x0) for x in [xM1, xM2]]
        roots_i = roots_dist.index(min(roots_dist))
        x0M = [xM1, xM2][roots_i]

    x0m = np.nan
    if (not no_result) and (not no_result_m):   
        xm1 = (-bm + sqrm**0.5) / (2 * cm)
        xm2 = (-bm - sqrm**0.5) / (2 * cm)

        roots_dist = [abs(x - x0) for x in [xm1, xm2]]
        roots_i = roots_dist.index(min(roots_dist))
        x0m = [xm1, xm2][roots_i]    

    if no_result:
        return np.nan, np.nan
    else:
        delta_x0 = np.nanmean([np.abs(x0-x0M), np.abs(x0-x0m)])
        return x0, delta_x0

def angular_dist(az1, az2):
    """
    Calculate the angular distance between two azimuth angles.

    Parameters:
    az1 (float): The first azimuth angle in degrees.
    az2 (float): The second azimuth angle in degrees.

    Returns:
    float: The angular distance between the two azimuth angles.
    """
    angular_distance_abs = abs(az1 - az2)
    return min(angular_distance_abs, 360 - angular_distance_abs)


def quadratic_func(x, a, b, c):
    return c * x**2 + b * x + a
