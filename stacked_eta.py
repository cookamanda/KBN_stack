# =============================================================================
# stacked_eta.py, Written by Amanda M. Cook 
# created: 2023-10-05
# last modified: 2023-10-24
# =============================================================================


#TODO: Prior is uniform right now --Amanda

import mpmath as mp
import numpy as np

import mpmath as mp
import numpy as np
from tqdm import tqdm 

def poisson(N, mu):
    return mp.fmul(mp.exp(mp.fneg(mu)), mp.fdiv(mp.power(mu, N), mp.factorial(N)))

def prior(eta_value):
    return 1

def likelihood_integrand(eta, S, N, B, F, F_err, F_2_S=3.799E-12):
    mean_xray_fluence = eta*F/F_2_S
    #propagate the radio error to x-ray error
    x_ray_fluence_err = eta*F_err/F_2_S
    #defining parameters which define the truncated norm (just defines where 0
    #is in terms of z score, not truncated from above (np.inf)
    return mp.fdiv(mp.fmul(poisson(N, (S+B)), mp.npdf(S, mu=mean_xray_fluence,
                                       sigma=x_ray_fluence_err )),
                   mp.nsum(lambda x: mp.npdf(x, mu=3,sigma=1), [0, mp.inf]))


def integrate_likelihood(eta, N, B, F, F_err, F_2_S= 7.631465e-12):
    result = mp.quad(lambda S: likelihood_integrand(eta, S, N, B, F, 
                                                    F_err, F_2_S), [0, mp.inf])
    return result

def stacked_eta_log_posterior(eta, N_i, B_i, F_i, F_err_i, F_2_S=7.631465e-12):
    log_integrals = mp.zeros(1,len(N_i))
    for i in range(len(N_i)):
        log_integrals[0,i] =  mp.log(integrate_likelihood(eta, N_i[i], B_i[i],
                                                        F_i[i],F_err_i[i], 
                                                        F_2_S))
    return mp.fadd(mp.log(prior(eta)), mp.fsum(log_integrals))

def credible_region(stacked_eta_posterior, etas,  level= 0.997,
                    return_indicies=False):
    #normalize
    exponentiated = [mp.exp(float(x)) for x in  stacked_eta_posterior] 
    normed = [mp.fdiv(mp.exp(float(x)),mp.fsum(exponentiated)) for x in exponentiated]
    RUL_i = np.argmin(np.abs(np.subtract(np.cumsum(normed), 1-((1-level)/2))))
    RLL_i = np.argmin(np.abs(np.subtract(np.cumsum(normed), ((1-level)/2))))
    i = 0 
    RLL = etas[RLL_i]
    RUL = etas[RUL_i]
    if return_indicies:
        return RLL, RUL, RLL_i, RUL_i
    else:
        return RLL, RUL

def eta_credible_region(N_i, B_i, F_i, F_i_err, max_eta=1E7, 
                        F_2_S = 7.631465e-12, nsamples=500, level=0.95,return_indicies = False):
    nsamples = int(nsamples)
    etas = np.logspace(0, np.log10(max_eta), int(nsamples)-1)
    etas_edges = np.logspace(0, np.log10(max_eta), int(nsamples))
    bin_widths = np.subtract(etas_edges[1:], etas_edges[:-1])
    log_posterior = mp.zeros(1,int(nsamples)-1)
    print('post_calc')
    for i, eta in tqdm(enumerate(etas)): 
        #print(eta)
        log_posterior[0,i] = mp.fadd(stacked_eta_log_posterior(eta, N_i, B_i, F_i, F_i_err,\
                      F_2_S), mp.log(bin_widths[i]))
        #print(log_posterior[i])
    if return_indicies:
        RLL, RUL, RLL_i, RUL_i = credible_region(log_posterior, etas, level, return_indicies=True) 
        return RLL, RUL, RLL_i, RUL_i
    else:
        RLL, RUL = credible_region(log_posterior, etas, level, return_indicies)                                                         
        return RLL, RUL

