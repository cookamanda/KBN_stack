import matplotlib.pyplot as plt
from etastack import *

prior_test = np.load('R1_XMM_posterior.npy')
N_i_test = [0,0,0,0,0,0,0] 
B_i_test = [5e-6, 5e-6, 5e-6, 5e-6,5e-6, 5e-6, 5e-6] 
F_i_test = np.multiply(200e6*1e-26,  [0.33, 0.83, 0.62, 1.08, 0.22, 0.37,  0.10])
F_i_err_test = np.linspace(0.005*1e-26*600e6,0.005*1e-26*600e6, 7)

F_2_S_test = [3.5E-11,3.5E-11,3.5E-11,3.5E-11,3.5E-11,3.5E-11,3.5E-11]
etas_test = np.load('R1_etas.npy')
eta_binwidths = np.load('R1_eta_binwidths.npy')
posterior_test = eta_posterior(etas_test,eta_binwidths, N_i_test, B_i_test, F_i_test, F_i_err_test, F_2_S_test, prior=prior_test)

plt.plot(etas_test, posterior_test)
plt.xscale('log')

cred_region = credible_region(posterior_test, etas_test, eta_binwidths, 0.997, return_tolerance=True)
plt.axvline(cred_region[1])
print(cred_region)
np.save('R1_chandra_posterior.npy',posterior_test)
np.save('R1_chandra_997_eta.npy', cred_region[1])
