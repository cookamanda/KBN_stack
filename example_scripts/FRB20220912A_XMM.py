import matplotlib.pyplot as plt
from etastack_HPDI import *


N_i_test = 0 #number of photons at the time of the radio burst
B_i_test = 0.0165 #avg background counts per searched burst width
F_i_test = (12.60845700*1e-26*180e6) #erg/cm/cm/s
F_i_err_test =(0.4*1e-26*200e6) #erg/cm/cm/s
F_2_S_test = 2.50794e-11 #(erg/cm/cm/s) / photon
eta_edges = np.logspace(0,9,1000) # defines the eta range you will be computing the posterior over
etas_test = eta_edges[:-1] + ((eta_edges[1:] - eta_edges[:-1])/2) # same as above, but going from bin edges to bin centers
eta_binwidths = (eta_edges[1:] - eta_edges[:-1]) #need widths for proper normalization
np.save('R117_eta_binwidths.npy',eta_binwidths) #saving these so they are consistent between all stacked posteriors
np.save('R117_etas.npy', etas_test) #saving these so they are consistent between all stacked posteriors
posterior_test = eta_posterior(etas_test,eta_binwidths, N_i_test, B_i_test, F_i_test, F_i_err_test, F_2_S_test, prior=None) #no (flat) prior here since it is the first I compute

plt.plot(etas_test, posterior_test)
plt.xscale('log')
cred_region = credible_region(posterior_test, etas_test, eta_binwidths, 0.997, return_tolerance=True)
print(cred_region) #will print lower bound, upper bound, and the difference between 0.997 and the computed region (tolerance)

np.save('R117_XMM_posterior.npy',posterior_test) #this is for plotting and to import as a prior for my next observations
np.save('R117_XMM_997_eta.npy', cred_region[1]) #this is for plotting 
