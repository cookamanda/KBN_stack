from sigpyproc.readers import FilReader
import pandas as pd
from astropy import time
from astropy import coordinates
from astropy import units
import numpy as np

def get_header(fil_file): #TODO: check to_dict method
    '''Load a filterbank header to a Python dict.'''
    
    header = {}
    sig_header = FilReader(fil_file).header
    header["tstart"] = float(sig_header.tstart)
    header["tsamp"] = float(sig_header.tsamp)
    header["nsamples"] = int(sig_header.nsamples)
    header["ra"] = str(sig_header.ra)
    header["dec"] = str(sig_header.dec)
    header["telescope"] = str(sig_header.telescope)
    
    return header

def topo_to_bary(topo_times, ra="23:09:04.9", dec="+48:42:25.4", site="CHIME"):
    '''Convert a list of MJD topocentric time to a list of MJD barycentric times.'''
    
    # determine time corrections and apply them to the topo times
    sky_coords = coordinates.SkyCoord(ra, dec, unit=(units.hourangle, units.deg), frame='icrs')
    telescope_site = coordinates.EarthLocation.of_site(site)  
    times = time.Time(topo_times, format='mjd', scale='utc', location=telescope_site)  
    time_corrections = times.light_travel_time(sky_coords).to_value("jd") 
    bary_times = topo_times - time_corrections #TODO: plus or minus??
    
    return bary_times

def samp_chooser(duration, tsamp, ds_factor):
    '''Get desired time sample factor the the nearest multiple of downsample factor.'''
    num_samps = round(duration / tsamp) + 1
    optimal_num_samps = num_samps / ds_factor
    if optimal_num_samps.is_integer():
        return optimal_num_samps
    else:
        optimal_num_samps = (round(optimal_num_samps) + 1) * ds_factor
    return optimal_num_samps