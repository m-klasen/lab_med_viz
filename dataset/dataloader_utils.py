import numpy as np
from monai import transforms as T
import nibabel as nib
from fastprogress.fastprogress import master_bar, progress_bar

import ctypes
import multiprocessing as mp


def scale_mri(image, q_min=1, q_max=99):
    image = np.clip(np.float32(image), *np.percentile(np.float32(image), [q_min, q_max]))
    image -= np.min(image)
    image /= np.max(image)
    return np.float32(image)

def scale_voxel_spacing(idx, img, segm, df, voxel_spacing):
        sample_voxel_spacing = np.array([df['x'],df['y'],df['z']])
        sample_vxsp = sample_voxel_spacing[:,idx]
        scale_factor = sample_vxsp / voxel_spacing
        
        img = zoom(img, scale_factor, order=3)
        segm = zoom(segm, scale_factor, order=3)
        return img,segm


    
def create_shared_arrays(CFG,df,root_dir):
    monai_T = T.Compose([T.CenterSpatialCropd(keys=("image","seg"),roi_size=CFG.img_size),
                         T.SpatialPadd(keys=("image","seg"),spatial_size=CFG.img_size)])
    img_paths = df['MRI_scaled_voxel_spacing']
    segm_paths = df['brain_mask_scaled_voxel_spacing']
        
    n_samples = len(df['id'])*(300// CFG.slice_sampling_interval)
    shared_array_base = mp.Array(ctypes.c_float, n_samples*256*256)
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(n_samples,256,256)

    shared_array_base2 = mp.Array(ctypes.c_float, n_samples*256*256)
    shared_array2 = np.ctypeslib.as_array(shared_array_base2.get_obj())
    shared_array2 = shared_array2.reshape(n_samples, 256,256)            

    curr_idx = 0
    for idx in progress_bar(range(len(df)),len(df)):
        img_fn = img_paths[idx]
        sgm_fn = segm_paths[idx]
        img = nib.load(f"{root_dir}/{img_fn}").get_fdata()
        segm = nib.load(f"{root_dir}/{sgm_fn}").get_fdata()  
        #img, segm = scale_voxel_spacing(idx, img, segm, df)
        img = scale_mri(img)

        #change from sideways orientation to top down
        img = np.transpose(img,(2,0,1))
        segm = np.transpose(segm,(2,0,1))

        tfmed = monai_T({'image':img, 'seg':segm})       
        img = tfmed['image']
        segm = tfmed['seg']
        
        #slice subsampling
        img = img[::CFG.slice_sampling_interval]
        segm = segm[::CFG.slice_sampling_interval]
        

        c = img.shape[0]
        shared_array[curr_idx:curr_idx+c] = img
        shared_array2[curr_idx:curr_idx+c] = segm
        curr_idx += c
    return shared_array,shared_array2