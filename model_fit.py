import matplotlib.pyplot as plt
import numpy as np
import inspect
from scipy.optimize import curve_fit 
from uncertainties.unumpy import uarray, nominal_values


def fit(f, x, y, y_std, p0, **kwg):
    """ 
    Fits y(x) +/- y_std data using scipy.optimize.curve_fit. Returns parameters with uncertainties and the resulting fit.
    
    f : callable
        The model function, f(x, …). It must take the independent variable as the first argument and the parameters to fit as separate remaining arguments.
        
    x : array-like
        x-coordinates of the data
    y : array-like 
        data
    y_std : array-like
        Standard deviations of "y" (data)
    p0 : array-like
        initial parameter guess
    **kwg : keyword params for scipy.optimize.least_squares
    """

    curve_fit_kwgs = dict(method='lm', absolute_sigma=True, **kwg)
    
    p, pcov = curve_fit(f, x, y, p0, sigma=y_std, **curve_fit_kwgs)

    ### Fitted y ###
    y_fit = f(x, *p)
    dx = x[1] - x[0]
    x_high_res = np.arange(x[0], x[-1] + 1e-2*dx, 1e-2*dx)
    y_fit_high_res = f(x_high_res, *p)
    ###----------------------------------------------------
    
    resid_norm = (y - y_fit)/y_std

    pars = {name: val for name, val in zip(list(inspect.signature(f).parameters.keys())[1:], 
                                           uarray(p, np.sqrt(np.diag(pcov))))}

    return dict(pars=pars, x=x, y=y, y_std=y_std, y_fit=y_fit, resid_norm=resid_norm, x_high_res=x_high_res, y_fit_high_res=y_fit_high_res)


def fit_with_bootstrap(f, x, y, y_std, p0, mc_trials, **kwg):
    """
    Fits y(x) +/- y_std data using scipy.optimize.curve_fit. Returns parameters with uncertainties and the resulting fit.
    
    f : callable
        The model function, f(x, …). It must take the independent variable as the first argument and the parameters to fit as separate remaining arguments.
        
    x : array-like
        x-coordinates of the data
    y : array-like 
        data
    y_std : array-like
        Standard deviations of "y" (data)
    p0 : array-like
        initial parameter guess
    mc_trials : number of trials (simulations) in the parameter Monte-Carlo 
    
    **kwg : keyword params for scipy.optimize.least_squares
    """

    fit0 = fit(f, x, y, y_std, p0, **kwg)

    ### Fitted y ###
    y_fit = fit0['y_fit']
    dx = x[1] - x[0]
    x_high_res = np.arange(x[0], x[-1] + 1e-2*dx, 1e-2*dx)
    y_fit_high_res = f(x_high_res, *p)
    ###----------------------------------------------------

    resid_norm = (y - y_fit)/y_std
    
    y_sim = np.random.normal(loc=fit0['fit'], scale=y_std, size=(n_trials, len(x)))
    
    pars_mc_full = np.array([nominal_values(fit(f, x, y_sim_i, y_std, nominal_values(fit0['pars']), **kwg)['pars']) for y_sim_i in y_sim])

    pars_mc = {name: val for name, val in zip(list(inspect.signature(f).parameters.keys())[1:], 
                                           uarray(np.mean(pars_mc_full, axis=0), np.std(pars_mc_full, axis=0)))}

    return dict(pars=pars_mc, x=x, y=y, y_std=y_std, y_fit=y_fit, resid_norm=resid_norm, x_high_res=x_high_res, y_fit_high_res=y_fit_high_res)


## Auxillary fucntions ###

def print_params(pars):

    for name, val in pars.items():
        print(f'{name} = {val}')


### Plotting ###

def plot_fit(fit, figsize=(8, 6), xlabel=None, ylabel=None, markersize=None, elinewidth=0.75):

    fig, [ax_fit, ax_res] = plt.subplots(2, 1, sharex=True, figsize=figsize)

    # ax_fit.plot(fit['x'], fit['y'], c='k', ls='', marker='.', markersize=markersize)
    ax_fit.errorbar(fit['x'], fit['y'], fit['y_std'], c='k', ls='', marker='.', markersize=markersize, elinewidth=elinewidth)
    ax_fit.plot(fit['x_high_res'], fit['y_fit_high_res'], c='r')

    ax_res.plot(fit['x'], fit['resid_norm'], c='k')

    ax_res.set_xlim(fit['x'][0], fit['x'][-1])
    ax_fit.set_xlabel('')
    ax_fit.set_ylabel(ylabel)
    ax_res.set_xlabel(xlabel)
    ax_res.set_ylabel('Standardized residuals (dim. less)')
    ax_res.axhline(y=0, ls='--', lw=1, c='k')
    
    plt.tight_layout()
