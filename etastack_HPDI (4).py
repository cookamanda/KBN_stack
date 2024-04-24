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
    implied_si = eta*F_i/F_2_S
    if isinstance(N_i, (int, float)):
        test_s = np.logspace(np.log10(implied_si)-5,
                             np.log10(implied_si)+5,
                             5000)
        bin_widths = np.logspace(np.log10(implied_si)-5,
                             np.log10(implied_si)+5,
                             5001)
        bin_edges = bin_widths[1:] - bin_widths[:-1]
        result = np.sum(np.multiply(S_i_integrand(test_s, N_i, B_i, a_TNorm, eta, F_i, F_2_S, F_i_err), bin_edges))
        return result
    else:
        test_s = np.logspace(np.log10(np.min(implied_si))-5,
                             np.log10(np.max(implied_si))+5,
                             5000)
        bin_widths = np.logspace(np.log10(np.min(implied_si))-5,
                             np.log10(np.max(implied_si))+5,
                             5001)
        bin_edges = bin_widths[1:] - bin_widths[:-1]
        result = np.zeros(len(N_i))
        for i in range(len(N_i)):
            result[i] =  np.sum(np.multiply(S_i_integrand(test_s, N_i[i], B_i[i], a_TNorm[i], eta, F_i[i], F_2_S[i], F_i_err[i]), bin_edges))
        return np.prod(result)   

def eta_posterior(etas, binsizes, N_i, B_i, F_i, F_i_err, F_2_S, prior=None):
    '''
    Prior, default is flat, same length as etas
    '''
    posterior = np.zeros(len(etas))
    if prior is None:
        prior = np.linspace(1.0/(max(etas)-min(etas)),1.0/(max(etas)-min(etas)),len(etas))
    for i, eta in tqdm(enumerate(etas)):
        a_TNorm = (0 - eta*F_i/F_2_S) / (F_i_err*eta/F_2_S)
        int_result = custom_integrate_S_i_function(N_i, B_i, a_TNorm, eta, F_i, F_2_S, F_i_err)
        posterior[i] = int_result*prior[i]
    norm = np.sum(np.multiply(posterior,binsizes))
    return np.divide(posterior,norm)


def credible_region(posterior, etas, binsizes, level, return_tolerance=False):
    multiplied_by_width = np.multiply(posterior, binsizes)
    eta_upper = etas[np.argmin(np.abs(np.subtract(np.cumsum(multiplied_by_width), 1-((1-level)))))]
    eta_lower = 0
    if return_tolerance==True:
        tolerance = min(np.abs(np.subtract(np.cumsum(multiplied_by_width), 1-((1-level)))))
        return eta_lower, eta_upper, tolerance
    else:
        return eta_lower, eta_upper

