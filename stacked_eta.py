# =============================================================================
# stacked_eta.py, Written by Amanda M. Cook 
# created: 2023-10-05
# last modified: 2023-10-05
# =============================================================================

import numpy as np 
from scipy.integrate import quad
from scipy.stats import poisson, truncnorm 
import math
from scipy.optimize import newton


def pois_rate_equation(lambda_, N, conf_lim = 0.95):
    result = 0
    for k in range(N + 1):
        result += (lambda_ ** k) * (math.exp(-lambda_)) / math.factorial(k)
    return result - (1-conf_lim)

def solve_rate(N):
    initial_guess = N+3*np.sqrt(N)  # You can start with an initial guess
    rate_solution = newton(pois_rate_equation, initial_guess, args=(N,))
    return rate_solution

def prior(eta_value):
    return 1

def likelihood_integrand(eta, S, N, B, F, F_err, F_2_S=3.799E-12):
    mean_xray_fluence = np.multiply(eta,F)/F_2_S
    #propagate the radio error to x-ray error
    x_ray_fluence_err = eta*F_err/F_2_S
    #defining parameters which define the truncated norm (just defines where 0
    #is in terms of z score, not truncated from above (np.inf))
    a_trunc, b_trunc = (0 - mean_xray_fluence ) / x_ray_fluence_err, np.inf
    return poisson.pmf(N, mu=(S+B)) * truncnorm.pdf(S, a_trunc, b_trunc, 
                                                    loc=mean_xray_fluence, 
                                                    scale=x_ray_fluence_err)

def integrate_likelihood(eta, N, B, F, F_err, F_2_S=3.799E-12, max_rate=10):
    integrand = lambda S: likelihood_integrand(eta, S, N, B, F, F_err, F_2_S)
    result, _ = quad(integrand, 0, max_rate)
    return result

def stacked_eta_log_posterior(eta, N_i, B_i, F_i, F_err_i, max_rate, F_2_S=3.799E-12):
    log_integrals = np.zeros(len(N_i))
    for i in range(len(N_i)):
        log_integrals[i] =  np.log(integrate_likelihood(eta, N_i[i], B_i[i],
                                                        F_i[i],F_err_i[i], 
                                                        F_2_S, max_rate))
    return np.add(np.log(prior(eta)), np.sum(log_integrals))

def credible_region(stacked_eta_posterior, etas,  level= 0.997):
    #normalize
    normed = np.divide(stacked_eta_posterior,
                       np.sum(np.exp(stacked_eta_log_posterior)))
    center_ind = np.argmax(normed)
    inc_conf = normed[int(center_ind)]
    i = 0 
    RLL = 0
    #if the most likely value is not 0
    if center_ind != 0:
        #add the posterior value of test rates until you've found your lower 
        #limit, i.e., until you've hit an integral value of conf_lim/2, or, you
        #hit a 0 rate
        while inc_conf < (level/2) and i <= (center_ind):
            i+= 1
            inc_conf += normed[int(center_ind-i)]
    
    
        RLL = etas[max(int(center_ind-i),0)]
    #from the lower limit, which might be 0, add rates to include in the 
    #credible region until the integral hits conf_lim 
    i = 1 
    while inc_conf < level:
        i+= 1
        inc_conf += normed[int(center_ind+i)]
    RUL = etas[int(center_ind+i)]
    return RLL, RUL

def eta_credible_region(N_i, B_i, F_i, F_i_err, min_eta=10E3, max_eta=10E6, 
                        F_2_S =  3.799E-12, nsamples=10000, level=0.997):
    max_rate = solve_rate(max(N_i))
    etas = np.logspace(np.log(min_eta), np.log(max_eta), nsamples)
    log_posterior = [stacked_eta_log_posterior(eta, N_i, B_i, F_i, F_i_err,\
                     max_rate, F_2_S) for eta in etas]
    RLL, RUL = credible_region(log_posterior, etas, level)                                                         
    return RLL, RUL

