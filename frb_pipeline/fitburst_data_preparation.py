import numpy as np

# Path to npz spectra array
npz_path = "/path/to/fit/input.npz"


# Used for configuration of internal arrays (not used for least-squares optimization)
metadata = { 
    "bad_chans"      : [],# a Python list of indices corresponding to frequency channels to zero-weight
    "freqs_bin0"     : 400.0,# a floating-point scalar indicating the value of frequency bin at index 0, in MHz
    "is_dedispersed" : False,# a boolean indicating if spectrum is already dedispersed (True) or not (False)
    "num_freq"       : 1024,# an integer scalar indicating the number of frequency bins/channels
    "num_time"       : 732422,# an integer scalar indicating the number of time bins
    "times_bin0"     : 59327.0,# a floating-point scalar indicating the value of time bin at index 0, in MJD
    "res_freq"       : 0.390625,# a floating-point scalar indicating the frequency resolution, in MHz
    "res_time"       : 40.96*1.0e-6,# a floating-point scalar indicating the time resolution, in seconds
}

# Used as initial guesses for least-squares optimization)
burst_parameters = {
    "amplitude"            : [1.0],# a list containing the the log (base 10) of the overall signal amplitude
    "arrival_time"         : [15.0],# a list containing the arrival times, in seconds
    "burst_width"          : [40.96*2*1.0e-6],# a list containing the temporal widths, in seconds
    "dm"                   : [87.757],# a list containing the dispersion measures (DM), in parsec per cubic centimeter
    "dm_index"             : [-2.0],# a list containing the exponents of frequency dependence in DM delay
    "ref_freq"             : [800.0],# a list containing the reference frequencies for arrival-time and power-law parameter estimates, in MHz (held fixed)
    "scattering_index"     : [-4.0],# a list containing the exponents of frequency dependence in scatter-broadening
    "scattering_timescale" : [0.0],# a list containing the scattering timescales, in seconds
    "spectral_index"       : [1.0],# a list containing the power-law spectral indices
    "spectral_running"     : [1.0],# a list containing the power-law spectral running
}

data_full = open(npz_path)


np.savez(
    "/path/to/fit/input/fitburst_input.npz", 
    data_full=data_full, 
    metadata=metadata, 
    burst_parameters=burst_parameters
)
