# =============================================================================
# stacked_eta.py, Written by Amanda M. Cook 
# created: 2023-10-05
# last modified: 2024-01-23
# =============================================================================


import numpy as np
from scipy.stats import poisson, truncnorm
from scipy.integrate import quad
from tqdm import tqdm
#posterior = 1/C p(η) ∑_i ∫_S_i Pois(N_i, B_i, F_i | λ = S_i) TNorm_0^∞(η F_i/(F_2_S), η F_i σ_F_i/(F_2_S)) d S_i

def S_i_integrand(S_i, N_i, B_i, a_TNorm, eta, F_i, F_2_S, F_i_err):
    log_poisson = poisson.logpmf(N_i, S_i + B_i)
    truncnorm_loc = eta * F_i / F_2_S
    truncnorm_scale = F_i_err * eta / F_2_S
    log_truncnorm = truncnorm.logpdf(S_i, a_TNorm, np.inf, loc=truncnorm_loc, scale=truncnorm_scale)
    return np.exp(np.float128(log_poisson + log_truncnorm))

def integrate_S_i_function(N_i, B_i, a_TNorm, eta, F_i, F_2_S, F_i_err):
    if isinstance(N_i, (int, float)):
        result, _ = quad(S_i_integrand, 0, np.inf, args=(N_i, B_i, a_TNorm, eta, F_i, F_2_S, F_i_err), epsabs=1e-18, epsrel=1e-18)
        return result
    else:
        result = np.zeros(len(N_i))
        for i in range(len(N_i)):
            result[i], _ = quad(S_i_integrand, 0, np.inf, args=(N_i[i], B_i[i], a_TNorm[i], eta, F_i[i], F_2_S[i], F_i_err[i]))
        return np.sum(result)

def custom_integrate_S_i_function(N_i, B_i, a_TNorm, eta, F_i, F_2_S, F_i_err):
    test_s = np.logspace(-5,2,1000)
    bin_widths = np.logspace(-5,2,1001)
    bin_edges = bin_widths[1:] - bin_widths[:-1]
    if isinstance(N_i, (int, float)):
        result = np.sum(np.multiply(S_i_integrand(test_s, N_i, B_i, a_TNorm, eta, F_i, F_2_S, F_i_err), bin_edges))
        return result
    else:
        result = np.zeros(len(N_i))
        for i in range(len(N_i)):
            result[i] =  np.sum(np.multiply(S_i_integrand(test_s, N_i[i], B_i[i], a_TNorm[i], eta, F_i[i], F_2_S[i], F_i_err[i]), bin_edges))
        return np.sum(result)   
    
def eta_posterior(etas, N_i, B_i, F_i, F_i_err, F_2_S, prior=None):
    '''
    Prior, default is flat, same length as etas
    '''
    posterior = np.zeros(len(etas))
    if prior is None:
        prior = np.linspace(1,1,len(etas))
    for i, eta in tqdm(enumerate(etas)):
        a_TNorm = (0 - eta*F_i/F_2_S) / (F_i_err*eta/F_2_S)
        int_result = custom_integrate_S_i_function(N_i, B_i, a_TNorm, eta, F_i, F_2_S, F_i_err)
        posterior[i] = int_result*prior[i]
    eta_edges = np.logspace(np.log10(min(etas)),np.log10(max(etas)), len(etas)+1)
    eta_binsizes = eta_edges[1:] - eta_edges[:-1]    
    norm = np.sum(np.multiply(posterior,eta_binsizes))
    return np.divide(posterior,norm)


def credible_region(posterior, etas, binsizes, level, return_tolerance=False):
    multiplied_by_width = np.multiply(posterior, binsizes)
    eta_upper = etas[np.argmin(np.abs(np.subtract(np.cumsum(multiplied_by_width), 1-((1-level)/2))))]
    eta_lower = etas[np.argmin(np.abs(np.subtract(np.cumsum(multiplied_by_width), (1-level)/2)))]
    if return_tolerance==True:
        tolerance = min(np.abs(np.subtract(np.cumsum(multiplied_by_width), 1-((1-level)/2))))
        return eta_lower, eta_upper, tolerance
    else:
        return eta_lower, eta_upper
            
