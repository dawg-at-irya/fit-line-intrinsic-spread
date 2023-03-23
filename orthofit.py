#Given an (x, y) pair with uncertainties (sx, sy), fit a line that minimises the orthogonal distance of points from the line.
#   Also incorporates a treatment of intrinsic scatter via V, the variance in this scatter ORTHOGONAL TO THE BEST-FIT LINE.
#   The program is written to maximise the log-likelihood in Equation 35 of Hogg et al. 2010.
import numpy as np
from scipy.optimize import minimize

def loglikelihood(pars, x, y, sigx, sigy):
    #Implement Equation 35 in Hogg et al. 2010.
    Delta = -x * np.sin(pars[0]) + y * np.cos(pars[0]) - pars[1]
    Sigma2 = sigx**2 * np.sin(pars[0])**2 + sigy**2 * np.cos(pars[0])**2
    logL = (np.log(Sigma2 + pars[2]) + Delta**2/(Sigma2 + pars[2])).sum()
    return 0.5*logL #Note positive sign

def orthofit(x, y, sigx, sigy, initial_guess = np.array([1., 1., 1.])):
    pars = minimize(loglikelihood, initial_guess, args = (x, y, sigx, sigy), \
                    options = {'maxiter': 10000}, method = "Nelder-Mead")
    slope = np.tan(pars['x'][0])
    intercept = pars['x'][1]/np.cos(pars['x'][0])
    V = pars['x'][2]
    if pars['status'] != 0:
        print("********ORTHOFIT did not succeed!")
        print(pars)
    return intercept, slope, V
