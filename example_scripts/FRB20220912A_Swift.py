import matplotlib.pyplot as plt
from etastack_HPDI_CHIMEPSRbursts import *
prior_test = np.load('R117_XMM_posterior.npy') #use the XMM posterior as our prior 
N_i_test = [0, 0, 0]  #number of photons at the time of the radio burst
B_i_test = [0.0015, 0.002, 0.002] #avg background counts per searched burst width
F_test_Jy_ms = [1.12489, 0.51784, 0.23914]
F_i_test = np.multiply(F_test_Jy_ms,1e-26*262.1e6) #erg/cm/cm/s
F_i_err_test = [] #these do nothing in the PSR burst version, modelled as a truncated Normal, where it is truncated at the rate implied by the 
#fluence lower limits from CHIME/PSR and the eta which is being tested, centered at 2* the implied rate, with std the implied rate
F_2_S_test = [1.93E-10, 1.93E-10, 1.99E-10] #photons / (erg/cm/cm/s) 
etas_test = np.load('R117_etas.npy') #load so they are the same as for the prior
eta_binwidths = np.load('R117_eta_binwidths.npy') #load so they are the same as for the prior
posterior_test = eta_posterior(etas_test,eta_binwidths, N_i_test, B_i_test, F_i_test, F_i_err_test, F_2_S_test, prior=prior_test)

plt.plot(etas_test, posterior_test)
plt.xscale('log')
cred_region = credible_region(posterior_test, etas_test, eta_binwidths, 0.997, return_tolerance=True) 
print(cred_region) #will print lower bound, upper bound, and the difference between 0.997 and the computed region (tolerance)
np.save('R117_Swift_997_eta.npy', cred_region[1])
np.save('R117_Swift_posterior.npy',posterior_test)
