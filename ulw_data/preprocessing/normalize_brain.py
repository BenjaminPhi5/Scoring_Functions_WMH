import numpy as np
import nibabel as nib

def normalize_brain(image, mask, lower_percentile=0, upper_percentile=100):
    image = image.copy()
    brain_locs = image[mask]
    
    if lower_percentile > 0 or upper_percentile < 100:
        brain_locs = brain_locs.flatten()
        sorted_indices = np.argsort(brain_locs)
        num_brain_voxels = len(sorted_indices)
        #print(num_brain_voxels)

        lower_index = int(lower_percentile*num_brain_voxels)
        upper_index = int(upper_percentile*num_brain_voxels)

        retained_indices = sorted_indices[lower_index:upper_index]
        #print(len(retained_indices)/num_brain_voxels)
        
        brain_locs = brain_locs[lower_index:upper_index]

    mean = np.mean(brain_locs)
    std = np.std(brain_locs)
    
    print(mean, std)

    image[mask] = (image[mask] - mean) / std

    return image

def nib_normalize_brain(nib_image, nib_mask, lower_percentile=0, upper_percentile=100):
    image_data = nib_image.get_fdata()
    mask_data = nib_mask.get_fdata().astype(bool)

    image_data = normalize_brain(image_data, mask_data)

    return nib.nifti1.Nifti1Image(image_data, affine=nib_image.affine, header=nib_image.header)


def get_brain_mean_std_without_mask(whole_img3D, cutoff=0.01):
    """
        get mean and starndard deviation of the brain pixels, 
        where brain pixels are all those pixels that are > cutoff 
        in intensity value.
        returns the mean, the std and the locations where the brain is present.
    """
    brain_locs = whole_img3D > cutoff # binary map, 1 for included
    brain3D = whole_img3D[brain_locs]
    
    mean = np.mean(brain3D)
    std = np.std(brain3D)
    
    return mean, std, brain_locs

def normalize_brain_without_mask(whole_img3D, cutoff=0.01):
    """
    whole_img3D: numpy array of a brain scan
    
    normalize brain pixels using global mean and std.
    only pixels > cutoff in intensity are included.
    """
    mean, std, brain_locs = get_brain_mean_std(whole_img3D, cutoff)
    whole_img3D[brain_locs] = (whole_img3D[brain_locs] - mean) / std

    return whole_img3D

def nib_normalize_brain_without_mask(nib_image, cutoff=0.01):
    image_data = nib_image.get_fdata()

    image_data = normalize_brain(image_data, cutoff)

    return nib.nifti1.Nifti1Image(image_data, affine=nib_image.affine, header=nib_image.header)
