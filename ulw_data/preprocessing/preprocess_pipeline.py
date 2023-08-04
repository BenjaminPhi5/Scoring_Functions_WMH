import os
from collections import defaultdict

from ulw_data.preprocessing.normalize_brain import *
from ulw_data.preprocessing.resample import *
from ulw_data.preprocessing.skull_strip import *
import nibabel as nib

def preprocess(file_dict, force=False, out_spacing=[1.,1.,3.], lower_norm_percentile=0, upper_norm_percentile=100):
    flair = file_dict['FLAIR']
    t1 = file_dict['T1']
    wmh = file_dict['wmh']
    out_path = file_dict['out_path']

    flair_out_path = f'{out_path}_FLAIR.nii.gz'
    t1_out_path = f'{out_path}_T1.nii.gz'
    mask_out_path = f'{out_path}_mask.nii.gz'
    wmh_out_path = f'{out_path}_wmh.nii.gz'

    # create a file used to let other processes
    # know this file is being processed.
    # a simple hack, allows me to run this script
    # across multiple unconnected efficiency, and
    # there are no race conditions here so doesn't
    # matter if two threads occasionally process the same file
    skip_file = f'{out_path}_skip_file.txt'
    if not force and os.path.exists(skip_file):
        print("skipping, file exists")
        return
        
    with open(skip_file, "w") as f:
            f.write(f"processing {out_path}")

    # resampling before we do anything else now, otherwise the hard thresholds put in place by the mask
    # can get lost at either end of the mask.
    print('# resample the flair')
    resample_and_save(flair, flair_out_path, is_label=False, out_spacing=out_spacing)
    
    print('# resample the T1')
    resample_and_save(t1, t1_out_path, is_label=False, out_spacing=out_spacing)
    
    print('# resample the wmh')
    resample_and_save(wmh, wmh_out_path, is_label=True, out_spacing=out_spacing)

    # compute the mask 
    print('# load files')
    t1 = nib.load(t1_out_path)
    flair = nib.load(flair_out_path)

    print('# compute mask and skull strip the T1')
    t1, mask = skull_strip(t1)
    nib.save(mask, mask_out_path)
    
    print('# skull strip the FLAIR')
    flair = apply_mask(flair, mask)
    
    print('# normalize the FLAIR')
    flair = nib_normalize_brain(flair, mask, lower_norm_percentile, upper_norm_percentile)
    nib.save(flair, flair_out_path)

    print('# normalize the T1')
    t1 = nib_normalize_brain(t1, mask, lower_norm_percentile, upper_norm_percentile)
    nib.save(t1, t1_out_path)
    
    print('# done!')