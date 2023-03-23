from scipy import odr
import numpy as np
from ss_setPlotParams import setPlotParams
from scipy.optimize import minimize
from orthofit import *
from urllib.request import urlopen

def get_excess_data():
    url = "http://www.irya.unam.mx/gente/s.srinivasan/Teaching/Statistics/python/datasets/Srinivasanetal2009_Crich_8micron_excess.csv"
    file = urlopen(url)
    data = np.loadtxt(file, skiprows = 1, dtype = {'names': ('L', 'sigma_L', 'X8', 'sigma_X8'), \
                                                   'formats': ('f4', 'f4', 'f4', 'f4')}, \
                      delimiter = ',')
    return data

def linear_fit(x, y, sy):
    #Invert the data matrices to get the best-fit parameter values and uncertainties
    ### Define A and Y matrices and the inverse of the covariance matrix
    A = np.array([np.ones(len(x)), x]).transpose()
    Y = np.array(y)[:, np.newaxis]
    Sigma_1 = np.linalg.inv(np.diag(sy**2))
    ### Compute best-fit parameters and parameter covariance matrix
    Temp = A.transpose().dot(Sigma_1)
    Sigma_Pars = np.linalg.inv(Temp.dot(A))
    X = Sigma_Pars.dot(Temp.dot(Y))
    return A, Y, X, Sigma_Pars

def fitfunc(B, x):
    return B[0] + B[1]*x

def main():
    data = get_excess_data()
    x = np.log10(data['L'])
    sx = data['sigma_L']/data['L']/np.log(10.)
    y = np.log10(data['X8'])
    sy = data['sigma_X8']/data['X8']/np.log(10.)

    #Simple linear fit to determine first guess for parameters
    _, _, parguess, _ = linear_fit(x, y, sy)
    parguess = parguess.flatten()

    #First, use Orthogonal Distance Regression
    myodr = odr.ODR(odr.RealData(x, y, sx = sx, sy = sy), odr.Model(fitfunc), beta0=parguess)
    odrout = myodr.run()
    print("ODR best-fit intercept and slope: {} and {}".format(np.round(odrout.beta[0], decimals = 2), \
                                                               np.round(odrout.beta[1], decimals = 2)))
    print("ODR unc. in intercept and slope: {} and {}".format(np.round(odrout.sd_beta[0], decimals = 2), \
                                                               np.round(odrout.sd_beta[1], decimals = 2)))

    #Now, use Hogg et al. algorithm to include orthogonal intrinsic spread
    #The initial guess for V is obtained from trial-and-error; the covariance matrix is computed for
    #   each value of V, then the value of V that brings the reduced chi-square close to 1 is selected.
    pars_init = np.array([np.arctan(parguess[1]), parguess[0]*np.cos(np.arctan(parguess[1])), 0.25])
    b, m, V = orthofit(x, y, sx, sy, initial_guess = pars_init)
    b_hi = b + np.sqrt(V)/(1/np.sqrt(1 + m**2))
    b_lo = b - np.sqrt(V)/(1/np.sqrt(1 + m**2))
    print("OVAR best-fit intercept and slope: {} and {}".format(np.round(b, decimals = 2), \
                                                               np.round(m, decimals = 2)))
    print("OVAR best-fit intrinsic variance: {}".format(np.round(V, decimals = 2)))

    #Generate grid for plot
    xx = np.linspace(np.min(x) * 0.8, np.max(x) * 1.2, 1000)
    yy_odr = odrout.beta[0] + odrout.beta[1] * xx
    yy_ovar = b + m * xx
    yy_ovar_hi = b_hi + m * xx
    yy_ovar_lo = b_lo + m * xx
    #Plot just the data
    plt = setPlotParams()
    plt.figure(figsize = (8, 8))
    plt.plot(x, y, 'o', color = 'black', label = "Data", ms = 0.6)
    plt.title(r'$8 \mu\textrm{m excess vs luminosity for LMC C-rich AGB stars}$')
    plt.xlabel(r"$\log{[\textrm{Luminosity} (L_\odot)]}$")
    plt.ylabel(r"$\log{[8 \mu\textrm{m Excess Flux (Jy)}]}$")
    plt.xlim(3., 5.5)
    plt.ylim(-4., -1.)
    plt.savefig('excess_data.png', bbox_inches = 'tight')
    #Plot data + ODR fit
    plt.plot(x, y, 'o', color = 'grey', label = "Data", ms = 0.6)
    plt = setPlotParams()
    plt.figure(figsize = (8, 8))
    plt.plot(x, y, 'o', color = 'grey', label = "Data", ms = 0.6)
    plt.plot(xx, yy_odr, linewidth = 3, color = 'blue', label = "ODR fit")
    plt.title(r'$8 \mu\textrm{m excess vs luminosity for LMC C-rich AGB stars}$')
    plt.xlabel(r"$\log{[\textrm{Luminosity} (L_\odot)]}$")
    plt.ylabel(r"$\log{[8 \mu\textrm{m Excess Flux (Jy)}]}$")
    plt.xlim(3., 5.5)
    plt.ylim(-4., -1.)
    plt.legend(loc = 'best')
    plt.savefig('excess_ODRfit.png', bbox_inches = 'tight')
    #Plot data + ODR AND OVAR fits
    plt.plot(x, y, 'o', color = 'grey', label = "Data", ms = 0.6)
    plt = setPlotParams()
    plt.figure(figsize = (8, 8))
    plt.plot(x, y, 'o', color = 'grey', label = "Data", ms = 0.6)
    plt.plot(xx, yy_odr, linewidth = 3, color = 'blue', label = "ODR fit")
    plt.plot(xx, yy_ovar, linewidth = 3, color = 'black', label = "OVAR fit (orthog. int. spread)")
    plt.plot(xx, yy_ovar_hi, linewidth = 3, color = 'black', label = "OVAR intercept + Sqrt(V)", ls = 'dashed')
    plt.plot(xx, yy_ovar_lo, linewidth = 3, color = 'black', label = "OVAR intercept - Sqrt(V)", ls = 'dashed')
    plt.title(r'$8 \mu\textrm{m excess vs luminosity for LMC C-rich AGB stars}$')
    plt.xlabel(r"$\log{[\textrm{Luminosity} (L_\odot)]}$")
    plt.ylabel(r"$\log{[8 \mu\textrm{m Excess Flux (Jy)}]}$")
    plt.xlim(3., 5.5)
    plt.ylim(-4., -1.)
    plt.legend(loc = 'best')
    plt.savefig('excess_ODR_and_OVARfit.png', bbox_inches = 'tight')

if __name__ == "__main__":
    main()

