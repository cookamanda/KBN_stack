from sigpyproc.readers import FilReader
import bpp_utilities as utils

# Input (burst TOAs from search pipeline and raw observation data)
burst_toas = [653.317898]
filpath = "/media/thomas/MARS7/CHIME/R117/R117_59880_pow.fil"

# Extract 30s around the burst, dedisperse and downsample
def burst_extractor(fil_file, burst_toas, duration=10, ds_factor=10, dm=219.456, write=False):
    '''Given a filterbank, extract smaller filterbanks with widths "extract_width" (in seconds)
       and centered around "burst_toas". Also downsample and dedisperse the extracts.'''
    
    # determine start and stop of extract in units of time samples
    num_samps = int(utils.samp_chooser(duration, utils.get_header(fil_file)["tsamp"], ds_factor))
    samp_start = int(round(burst_toas / utils.get_header(fil_file)["tsamp"]) - num_samps / 2)
    samp_end = int(samp_start + num_samps)
    print("Number of samples in extract: ", num_samps)

    # ensure the extract in contained within the observation
    if samp_start < 0:
        samp_start = 1
        print("Chosen interval preceeds observation time -> using start of observation instead")
    if samp_end > utils.get_header(filpath)["nsamples"]:
        samp_end = utils.get_header(filpath)["nsamples"]
        print("Chosen interval exceeds observation time -> using end of observation instead")

    # extract and dedisperse
    fil = FilReader(fil_file).read_dedisp_block(start=samp_start, nsamps=num_samps, dm=219.456)

    # downsample and save
    downsample_path = "/media/thomas/MARS7/CHIME/R117/R117_59880_pow_burst_extract.fil"
    fil = fil.downsample(tfactor=1)
    if write:
        fil.write_to_fil(downsample_path)
    
    return fil

def clean_fil(filpath):
    rfi_mask = FilReader(filpath).clean_rfi()
    return rfi_mask

# extract the burst
extracted_burst = burst_extractor(filpath, burst_toas, duration=10, ds_factor=10, dm=219.456, write=False)

# write the extracted burst to disk
np.save("/save/path/for/fit/input.npy", extracted_burst) # for ACF fit
#np.savez("/save/path/for/fit/input.npz", extracted_burst) # for fitburst data prep

