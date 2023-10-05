#stacked lim
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import quad
from tqdm import tqdm
from scipy.stats import poisson, truncnorm, norm
import math
from scipy.optimize import newton

def equation(lambda_, N, conf_lim = 0.95):
    result = 0
    for k in range(N + 1):
        result += (lambda_ ** k) * (math.exp(-lambda_)) / math.factorial(k)
    return result - (1-conf_lim)

def solve_rate(N):
    initial_guess = N+3*np.sqrt(N)  # You can start with an initial guess
    rate_solution = newton(equation, initial_guess, args=(N,))
    return rate_solution

def conf_intv(N_i, B_i, n_samples=1000, conf_lim=0.95):
    #P(obs counts geq N) = sum_k=N^inf lambda^k e^(-lambda)/(k!)
    #P(obs counts geq N) = 1 - sum_(k=0)^N lambda^K e^(-lambda) / k!
    #whats the max relevant rate to sample out to? Probably like 95% rate U.L. of the total obs 
    #counts. We can calculate that by P = 1 - sum_(k=0)^N lambda^k e^(-lambda)/ k! solve for
    #lambda. I'll write functions above
    max_rate = solve_rate(np.max(N_i))
    rates = np.linspace(0,max_rate,n_samples)
    S_levels = np.zeros(n_samples)
    prior_vals = np.zeros(n_samples)
    
    #a_trunc, b_trunc = (0 - prior_rate) / np.sqrt(prior_rate), np.inf
    for i, S in enumerate(rates):
        S_levels[i] = poisson.pmf(N_i, np.add(S,B_i)).prod()
        #prior_vals[i] = truncnorm.pdf(S, a_trunc, b_trunc, loc=prior_rate, scale=np.sqrt(prior_rate))
        prior_vals[i] = 1
    #compare percentiles
    posterior_density = S_levels*prior_vals/np.multiply(S_levels, prior_vals).sum()
    center_ind = np.argmax(posterior_density)
    inc_conf = posterior_density[int(center_ind)]
    i = 0 
    RLL = 0
    RLL_i = 0
    if center_ind != 0:
        while inc_conf < (conf_lim/2) and i <= (center_ind):
            i+= 1
            inc_conf += posterior_density[int(center_ind-i)]
    
    
        RLL = rates[max(int(center_ind-i),0)]
        RLL_i = max(int(center_ind-i),0)

    i = 1 
    while inc_conf < (conf_lim):
        i+= 1
        inc_conf += posterior_density[int(center_ind+i)]
    RUL = rates[int(center_ind+i)]
    RUL_i = int(center_ind+i)
    return RLL, RUL
