import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy import stats
from scipy.signal import correlate2d
from scipy.optimize import curve_fit, leastsq
from lmfit import minimize, Parameters, fit_report, Model



# Assume a normal/Gaussian distribution with mu = 0 and std. dev. = 1.
# Calculate the two-sided probability, given sigma.
def calc_confidence_limit_probability(sigma):

    # See: https://books.google.ca/books?id=sUymEAAAQBAJ&pg=PA386&lpg=PA386&dq=%22stats.norm.cdf(1.0)
    confidence_limit = stats.norm.cdf(sigma) - stats.norm.cdf(-sigma)

    # Alternatively: confidence_limit = scipy.special.erf(sigma / np.sqrt(2.0))
    # Alternatively: confidence_limit = (stats.norm.cdf(sigma) * 2.0) - 1.0

    return confidence_limit

# Need to add center parameter.
def lorentzian(x, gamma, a, c):
    return (a * gamma) / ((x**2.) + (gamma**2.)) + c

def gaussian(x, a, x0, sigma, c):
    return a * np.exp(-(x - x0)**2. / (2.0 * sigma**2.)) + c

def gaussian_fit(params, x_data, data):

    a = params['a']
    x0 = params['x0']
    sigma = params['sigma']
    c = params['c']

    fit = gaussian(x_data, a, x0, sigma, c)

    resid = data.ravel()-fit

    return resid

def gaussian_2d(x_data_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta):

    (x, y) = x_data_tuple
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2.)/(2*sigma_x**2.) + (np.sin(theta)**2.)/(2.*sigma_y**2.)
    b = -(np.sin(2.*theta))/(4.*sigma_x**2.) + (np.sin(2.*theta))/(4.*sigma_y**2.)
    c = (np.sin(theta)**2.)/(2.*sigma_x**2.) + (np.cos(theta)**2.)/(2.*sigma_y**2.)
    g = amplitude * np.exp(-(a*((x-xo)**2.) + 2.*b*(x-xo)*(y-yo) + c*((y-yo)**2.)))

    return g.ravel()

def gaussian_2d_fit(params, x_data_tuple, data):

    amplitude = params['amplitude']
    xo = params['xo']
    yo = params['yo']
    sigma_x = params['sigma_x']
    sigma_y = params['sigma_y']
    theta = params['theta']

    fit = gaussian_2d(x_data_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta)

    resid = data.ravel()-fit

    return resid

def autocorr_2D(arr):
    """
    Given an array arr, compute the ACF of the array
    """
    ACF = correlate2d(arr, arr)

    return ACF

def generate_acf_2d(ds, timeres, freqres, generate_acf_2d_flag, output_fn="", input_fn=""):

    if (generate_acf_2d_flag == True):
        ACF_raw = autocorr_2D(ds)
        np.savez(output_fn, ACF_raw=ACF_raw)

    elif (generate_acf_2d_flag == False):
        data = np.load(input_fn)
        ACF_raw = data["ACF_raw"]

    ACF_raw = ACF_raw / np.max(ACF_raw)
    # Mask the zero-lag spike.
    ACF = np.ma.masked_where(ACF_raw == np.max(ACF_raw), ACF_raw)

    ACFtime = np.sum(ACF_raw, axis=0)
    ACFfreq = np.sum(ACF_raw, axis=1)
    ACFtime = np.ma.masked_where(ACFtime == np.max(ACFtime), ACFtime)
    ACFfreq = np.ma.masked_where(ACFfreq == np.max(ACFfreq), ACFfreq)

    # Make the time and frequency axes
    time_one = np.arange(1., ds.shape[1], 1.) * timeres * 1000.  # ms
    times = np.concatenate((-time_one[::-1], np.concatenate(([0.], time_one))))
    freq_one = np.arange(1., ds.shape[0], 1.) * freqres
    freqs = np.concatenate((-freq_one[::-1], np.concatenate(([0.], freq_one))))

    # 1D Gaussian fitting to ACFtime and ACFfreq
    params_ACFtime = Parameters()
    params_ACFtime.add('a', value=1.)
    params_ACFtime.add('x0', value=0.)
    params_ACFtime.add('sigma', value=np.max(times))
    params_ACFtime.add('c', value=0.)

    ACFtime_fit_output = minimize(gaussian_fit, params_ACFtime, method='leastsq',
                                  kws={"x_data": times, "data": ACFtime})
    print("*** Gaussian fit to 1D ACFtime ***")
    print("Times (x) are in milliseconds")
    print(fit_report(ACFtime_fit_output))
    print("\n\n")

    params_ACFfreq = Parameters()
    params_ACFfreq.add('a', value=1.)
    params_ACFfreq.add('x0', value=0.)
    params_ACFfreq.add('sigma', value=np.max(freqs))
    params_ACFfreq.add('c', value=0.)

    ACFfreq_fit_output = minimize(gaussian_fit, params_ACFfreq, method='leastsq',
                                  kws={"x_data": freqs, "data": ACFfreq})
    print("*** Gaussian fit to 1D ACFtime ***")
    print("Frequencies (y) are in MHz")
    print(fit_report(ACFfreq_fit_output))
    print("\n\n")



    ACFtime_fit = gaussian(times, ACFtime_fit_output.params["a"], ACFtime_fit_output.params["x0"],
                           ACFtime_fit_output.params["sigma"], ACFtime_fit_output.params["c"])
    ACFfreq_fit = gaussian(freqs, ACFfreq_fit_output.params["a"], ACFfreq_fit_output.params["x0"],
                           ACFfreq_fit_output.params["sigma"], ACFfreq_fit_output.params["c"])

    # 2D Gaussian fitting
    times_m, freqs_m = np.meshgrid(times, freqs)
    times_m = times_m.astype('float64')
    freqs_m = freqs_m.astype('float64')

    # Defining the parameters
    params_ACF = Parameters()
    params_ACF.add('amplitude', value=1.)
    params_ACF.add('xo', value=0., vary=False)
    params_ACF.add('yo', value=0., vary=False)
    params_ACF.add('sigma_x', value=ACFtime_fit_output.params["sigma"].value,
                   min=(ACFtime_fit_output.params["sigma"].value - (0.5 * ACFtime_fit_output.params["sigma"].value)),
                   max=(ACFtime_fit_output.params["sigma"].value + (0.5 * ACFtime_fit_output.params["sigma"].value)))
    params_ACF.add('sigma_y', value=ACFfreq_fit_output.params["sigma"].value,
                   min=(ACFfreq_fit_output.params["sigma"].value - (0.5 * ACFfreq_fit_output.params["sigma"].value)),
                   max=(ACFfreq_fit_output.params["sigma"].value + (0.5 * ACFfreq_fit_output.params["sigma"].value)))
    params_ACF.add('theta', value=0.)

    ACF_fit_output = minimize(gaussian_2d_fit, params_ACF, method='leastsq',
                              kws={"x_data_tuple": (times_m, freqs_m), "data": ACF})
    print("*** Gaussian fit to 2D ACF ***")
    print("Times (x) are in milliseconds and Frequencies (y) are in MHz")
    print(fit_report(ACF_fit_output))
    print("\n\n")

    ACF_fit = gaussian_2d((times_m, freqs_m),
                          ACF_fit_output.params['amplitude'], ACF_fit_output.params['xo'], ACF_fit_output.params['yo'],
                          ACF_fit_output.params['sigma_x'], ACF_fit_output.params['sigma_y'], ACF_fit_output.params['theta'])
    ACF_fit = ACF_fit.reshape(len(freqs), len(times))

    return ACF, times, freqs, ACFtime, ACFfreq, ACFtime_fit, ACFfreq_fit, ACF_fit, ACFtime_fit_output,\
           ACFfreq_fit_output, ACF_fit_output



def plot_acf_2d(ACF, times, freqs, ACFtime, ACFfreq, ACFtime_fit, ACFfreq_fit, ACF_fit, ACFtime_fit_output,
                ACFfreq_fit_output, ACF_fit_output):

    plt.figure()
    plt.imshow(ACF, aspect="auto")
    plt.show(block=True)



    fig, ax = plt.subplots(2)
    ax[0].plot(times, ACFtime)
    ax[0].plot(times, ACFtime_fit)
    ax[0].axvline(ACFtime_fit_output.params["x0"] - ACFtime_fit_output.params["sigma"])
    ax[0].axvline(ACFtime_fit_output.params["x0"] + ACFtime_fit_output.params["sigma"])
    ax[0].set_ylabel('ACF time')
    ax[1].plot(freqs, ACFfreq)
    ax[1].plot(freqs, ACFfreq_fit)
    ax[1].axvline(ACFfreq_fit_output.params["x0"] - ACFfreq_fit_output.params["sigma"])
    ax[1].axvline(ACFfreq_fit_output.params["x0"] + ACFfreq_fit_output.params["sigma"])
    ax[1].set_ylabel('ACF freq')
    plt.show(block=True)



    # residuals
    ACFtime_fit = np.sum(ACF_fit, axis=0)
    ACFfreq_fit = np.sum(ACF_fit, axis=1)

    ACFtime_resid = ACFtime - ACFtime_fit
    ACFfreq_resid = ACFfreq - ACFfreq_fit

    # plot
    fig = plt.figure(figsize=(8, 8))
    rows = 3
    cols = 3
    widths = [3, 1, 1]
    heights = [1, 1, 3]
    gs = gridspec.GridSpec(ncols=cols, nrows=rows, width_ratios=widths, height_ratios=heights, wspace=0.0, hspace=0.0)

    cmap = plt.cm.viridis

    ax1 = fig.add_subplot(gs[0, 0])  # Time ACF
    min_ACFtime = np.min(ACFtime)
    norm_ACFtime = np.max(ACFtime - min_ACFtime)
    ACFtime_plot = ACFtime - min_ACFtime
    ACFtime_plot = ACFtime_plot / norm_ACFtime
    ax1.plot(times, ACFtime_plot, color='k')
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), visible=False)
    ax1.set_xlim(times[0], times[-1])
    ax1.axhline(0.5)
    ACFtime_fit_plot = ACFtime_fit / norm_ACFtime
    ACFtime_fit_plot = ACFtime_fit_plot - (min_ACFtime / norm_ACFtime)
    ax1.plot(times, ACFtime_fit_plot, color='purple')
    ax1.axis(ymin=-0.1, ymax=1.1)

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)  # Time ACF residuals
    ax2.plot(times, ACFtime_resid, color='k')
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.set_xlim(times[0], times[-1])

    ax3 = fig.add_subplot(gs[2, 0], sharex=ax2)  # 2D ACF
    T, F = np.meshgrid(times, freqs)
    ax3.imshow(ACF, aspect='auto', interpolation='None', origin='upper', cmap=cmap,
               extent=(times[0], times[-1], freqs[0], freqs[-1]))

    # Algorithm to determine the sigma levels for plotting contours.
    min_ACF_fit = np.min(ACF_fit)
    ACF_fit_rescaled = ACF_fit - min_ACF_fit
    normalization_ACF_fit = np.sum(ACF_fit_rescaled)
    ACF_fit_rescaled = ACF_fit_rescaled / normalization_ACF_fit

    ACF_fit_rescaled = ACF_fit_rescaled.flatten()
    ACF_fit_rescaled = np.sort(ACF_fit_rescaled)

    cl_1sigma = calc_confidence_limit_probability(1.0)
    cl_2sigma = calc_confidence_limit_probability(2.0)
    cl_3sigma = calc_confidence_limit_probability(3.0)
    cl_4sigma = calc_confidence_limit_probability(4.0)

    level_1sigma = 1. - cl_1sigma
    level_2sigma = 1. - cl_2sigma
    level_3sigma = 1. - cl_3sigma
    level_4sigma = 1. - cl_4sigma

    level_1sigma_value = 0.
    level_2sigma_value = 0.
    level_3sigma_value = 0.
    level_4sigma_value = 0.

    level_sum = 0.

    # Brute-force find the sigma levels.
    for i in np.arange(0, len(ACF_fit_rescaled), 1):
        level_sum = level_sum + ACF_fit_rescaled[i]

        if ((level_sum >= level_4sigma) and (level_4sigma_value == 0.)):
            level_4sigma_value = ACF_fit_rescaled[i]
            print(level_sum, level_4sigma, level_4sigma_value)

        if ((level_sum >= level_3sigma) and (level_3sigma_value == 0.)):
            level_3sigma_value = ACF_fit_rescaled[i]
            print(level_sum, level_3sigma, level_3sigma_value)

        if ((level_sum >= level_2sigma) and (level_2sigma_value == 0.)):
            level_2sigma_value = ACF_fit_rescaled[i]
            print(level_sum, level_2sigma, level_2sigma_value)

        if ((level_sum >= level_1sigma) and (level_1sigma_value == 0.)):
            level_1sigma_value = ACF_fit_rescaled[i]
            print(level_sum, level_1sigma, level_1sigma_value)

    level_1sigma_value = level_1sigma_value * normalization_ACF_fit
    level_1sigma_value = level_1sigma_value + min_ACF_fit

    level_2sigma_value = level_2sigma_value * normalization_ACF_fit
    level_2sigma_value = level_2sigma_value + min_ACF_fit

    level_3sigma_value = level_3sigma_value * normalization_ACF_fit
    level_3sigma_value = level_3sigma_value + min_ACF_fit

    level_4sigma_value = level_4sigma_value * normalization_ACF_fit
    level_4sigma_value = level_4sigma_value + min_ACF_fit



    ax3.contour(T, F, np.flipud(ACF_fit),
                [level_4sigma_value, level_3sigma_value, level_2sigma_value, level_1sigma_value],
                colors='r', linewidths=.5)

    ax3.set_ylabel('Freq lag (MHz)')
    ax3.set_xlabel('Time lag (ms)')
    #ax3.set_ylim([-10, 10])

    ax4 = fig.add_subplot(gs[2, 1], sharey=ax3)  # Freq ACF residuals
    ax4.plot(ACFfreq_resid, freqs, color='k')
    plt.setp(ax4.get_yticklabels(), visible=False)
    plt.setp(ax4.get_xticklabels(), visible=False)
    ax4.set_ylim(freqs[0], freqs[-1])
    #ax4.set_ylim([-10, 10])
    print(freq[0], freq[500])

    ax5 = fig.add_subplot(gs[2, 2], sharey=ax4)  # Freq ACF
    min_ACFfreq = np.min(ACFfreq)
    norm_ACFfreq = np.max(ACFfreq - min_ACFfreq)
    ACFfreq_plot = ACFfreq - min_ACFfreq
    ACFfreq_plot = ACFfreq_plot / norm_ACFfreq
    ax5.plot(ACFfreq_plot, freqs, color='k')
    plt.setp(ax5.get_xticklabels(), visible=False)
    plt.setp(ax5.get_yticklabels(), visible=False)
    ax5.set_ylim(freqs[0], freqs[-1])
    #ax5.set_ylim([-10, 10])
    ax5.axvline(0.5)
    ACFfreq_fit_plot = ACFfreq_fit / norm_ACFfreq
    ACFfreq_fit_plot = ACFfreq_fit_plot - (min_ACFfreq / norm_ACFfreq)
    ax5.plot(ACFfreq_fit_plot, freqs, color='purple')
    ax5.axis(xmin=-0.1, xmax=1.1)



    time_width = np.abs(2. * np.sqrt(2. * np.log(2.)) * ACFtime_fit_output.params["sigma"]) / np.sqrt(2.)
    freq_width = np.abs(2. * np.sqrt(2. * np.log(2.)) * ACFfreq_fit_output.params["sigma"]) / np.sqrt(2.)

    time_width_err = np.abs(2. * np.sqrt(2. * np.log(2.)) * ACFtime_fit_output.params["sigma"].stderr) / np.sqrt(2.)
    freq_width_err = np.abs(2. * np.sqrt(2. * np.log(2.)) * ACFfreq_fit_output.params["sigma"].stderr) / np.sqrt(2.)

    print("\n")
    print("Width (time, ms): %.20f +/- %.20f" % (time_width, time_width_err))
    print("Width (freq, MHz): %.20f +/- %.20f" % (freq_width, freq_width_err))

    time_fwhm = time_width * np.sqrt(2.)
    freq_fwhm = freq_width * np.sqrt(2.)

    ax1.axvline(ACFtime_fit_output.params["x0"] - (time_fwhm / 2.))
    ax1.axvline(ACFtime_fit_output.params["x0"] + (time_fwhm / 2.))
    ax5.axhline(ACFfreq_fit_output.params["x0"] - (freq_fwhm / 2.))
    ax5.axhline(ACFfreq_fit_output.params["x0"] + (freq_fwhm / 2.))

    ax3.axvline(ACFtime_fit_output.params["x0"] - (time_fwhm / 2.))
    ax3.axvline(ACFtime_fit_output.params["x0"] + (time_fwhm / 2.))
    ax3.axhline(ACFfreq_fit_output.params["x0"] - (freq_fwhm / 2.))
    ax3.axhline(ACFfreq_fit_output.params["x0"] + (freq_fwhm / 2.))

    plt.show(block=True)






ds = np.load("/path/to/fit/input.npy")

dt = 0.00016277999999999987
time = np.arange(0, np.shape(ds)[1], 1) * dt
df = (920.0 - 680.0) / np.shape(ds)[0]
freq = np.arange(680.0, 920.0, df)

idx = np.where((time >= 0.05) & (time <= 0.07))[0]
time = time[idx]
ds = ds[:, idx]

plt.figure()
plt.imshow(ds, aspect="auto", interpolation="None", origin="upper")
# plt.imshow(np.flipud(ds), aspect="auto", interpolation="None", origin="lower")
plt.show(block=True)


generate_acf_2d_flag = True   # True if you want to generate a new ACF from scratch
plot_acf_2d_flag = True

output_fn = "acf_fit_output.npz"
input_fn = output_fn

if (generate_acf_2d_flag == True):
    ACF, times, freqs, ACFtime, ACFfreq, ACFtime_fit, ACFfreq_fit, ACF_fit, ACFtime_fit_output, \
    ACFfreq_fit_output, ACF_fit_output = generate_acf_2d(ds, dt, df, generate_acf_2d_flag, output_fn=output_fn)

elif (generate_acf_2d_flag == False):
    ACF, times, freqs, ACFtime, ACFfreq, ACFtime_fit, ACFfreq_fit, ACF_fit, ACFtime_fit_output, \
    ACFfreq_fit_output, ACF_fit_output = generate_acf_2d(ds, dt, df, generate_acf_2d_flag, input_fn=input_fn)

if (plot_acf_2d_flag == True):
    plot_acf_2d(ACF, times, freqs, ACFtime, ACFfreq, ACFtime_fit, ACFfreq_fit, ACF_fit, ACFtime_fit_output,
                ACFfreq_fit_output, ACF_fit_output)
