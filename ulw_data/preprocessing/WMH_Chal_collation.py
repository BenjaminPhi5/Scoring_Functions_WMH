"""
# TODO: turn this into a proper script so that I can run it on the cluster. However, for now on to other things.
"""
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
from ulw_data.preprocessing.WMH_Chal_file_parser import get_files_dict
from collections import defaultdict
from tqdm import tqdm

preprocessed_path = "/media/benp/NVMEspare/datasets/full_WMH_Chal_dataset/preprocessed/individual_files"
ds_path = "/media/benp/NVMEspare/datasets/full_WMH_Chal_dataset"
collated_path = "/media/benp/NVMEspare/datasets/full_WMH_Chal_dataset/preprocessed/collated"

# split up each entry by it's domain and its fold (training or test)
files_dict = get_files_dict(ds_path, preprocessed_path)
files_dict_items_by_domain = defaultdict(lambda: {})
for key, entry in files_dict.items():
    files_dict_items_by_domain[f"{entry['domain']}_{entry['fold']}"][key] = entry

def load_data(subject_entry):
    subject_path = subject_entry['out_path'] + '_'
    flair = subject_path + 'FLAIR.nii.gz'
    t1 = subject_path + 'T1.nii.gz'
    mask = subject_path + 'mask.nii.gz'
    wmh = subject_path + 'wmh.nii.gz'

    flair = nib.load(flair).get_fdata()
    t1 = nib.load(t1).get_fdata()
    mask = nib.load(mask).get_fdata()
    wmh = nib.load(wmh).get_fdata()

    #print(subject_path, flair.shape, t1.shape, mask.shape, wmh.shape)

    data = np.stack([flair, t1, mask, wmh], axis=0)

    return data

def calculate_dim_pad(arr_size, target_size):
    assert target_size > arr_size

    diff = target_size - arr_size
    if diff % 2 == 0:
        pad = (diff//2, diff//2)
    else:
        pad = (diff//2, diff//2+1)

    return pad

def crop_and_pad_brain(image, out_shape=[192,224,64]):
    # channels are: flair, t1, mask, wmh.
    mask = image[2]
    brain_locs = np.where(mask)

    # crop
    image = image[
        :,
        brain_locs[0].min():brain_locs[0].max(),
        brain_locs[1].min():brain_locs[1].max(),
        brain_locs[2].min():brain_locs[2].max(),    
    ]

    # pad
    shape = image.shape
    pad0 = calculate_dim_pad(shape[1], out_shape[0])
    pad1 = calculate_dim_pad(shape[2], out_shape[1])
    pad2 = calculate_dim_pad(shape[3], out_shape[2])
    image = np.pad(image, ((0,0),pad0, pad1, pad2))

    return image
    


try:
    os.makedirs(collated_path)
except FileExistsError:
    print("Warning: collation folder already exists. Continuing")


for domain, domain_entries in files_dict_items_by_domain.items():
    domain_data = []
    domain_ids = []
    print(f"processing files from {domain}")
    # load the data
    for subject_id, entry in tqdm(domain_entries.items(), position=0, leave=True):
        domain_data.append(load_data(entry))
        domain_ids.append(subject_id)

    # crop and pad each image
    domain_data = [crop_and_pad_brain(data, out_shape=[192,224,64]) for data in domain_data]

    # calculate the range of the mask bounds for each image...
    
    np.savez_compressed(
        file=f'{collated_path}{os.path.sep}{domain}.npz',
        args=domain_data,
        kwds=domain_ids
    )
    
