import numpy as np
from monai import transforms as T
import nibabel as nib
from fastprogress.fastprogress import master_bar, progress_bar

import ctypes
import multiprocessing as mp


def scale_mri(image, percentile):
    if percentile:
        image = np.clip(np.float32(image), *np.percentile(np.float32(image), [percentile[0], percentile[1]]))
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
    img_size = CFG.img_size[0]
    monai_T = T.Compose([T.CenterSpatialCropd(keys=("image","seg"),roi_size=CFG.img_size),
                         T.SpatialPadd(keys=("image","seg"),spatial_size=CFG.img_size)])
    img_paths = df[CFG.img_paths]
    segm_paths = df[CFG.segm_paths]
        
    n_samples = len(df['id'])*int((np.ceil(256/CFG.slice_sampling_interval)))
    shared_array_base = mp.Array(ctypes.c_float, n_samples*img_size*img_size)
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(n_samples,img_size,img_size)

    shared_array_base2 = mp.Array(ctypes.c_float, n_samples*img_size*img_size)
    shared_array2 = np.ctypeslib.as_array(shared_array_base2.get_obj())
    shared_array2 = shared_array2.reshape(n_samples,img_size,img_size)            

    curr_idx = 0
    for idx in progress_bar(range(len(df)),len(df)):
        img_fn = img_paths[idx]
        sgm_fn = segm_paths[idx]
        img = nib.load(f"{root_dir}/{img_fn}").get_fdata()
        segm = nib.load(f"{root_dir}/{sgm_fn}").get_fdata()  
        #img, segm = scale_voxel_spacing(idx, img, segm, df)
        if CFG.scale_mri:
            img = scale_mri(img, CFG.percentile)

        #change from sideways orientation to top down
        img = np.transpose(img,CFG.transpose)[None,:,:,:]
        segm = np.transpose(segm,CFG.transpose)[None,:,:,:]

        tfmed = monai_T({'image':img, 'seg':segm})       
        img = tfmed['image'].squeeze()
        segm = tfmed['seg'].squeeze()
        
        #slice subsampling
        img = img[::CFG.slice_sampling_interval]
        segm = segm[::CFG.slice_sampling_interval]
        

        c = img.shape[0]
        shared_array[curr_idx:curr_idx+c] = img
        shared_array2[curr_idx:curr_idx+c] = segm
        curr_idx += c
    return shared_array,shared_array2


def create_shared_fcm_masks(CFG,df,root_dir):
    img_size = CFG.img_size[0]
    monai_T = T.Compose([T.CenterSpatialCropd(keys=("image"),roi_size=CFG.img_size),
                         T.SpatialPadd(keys=("image"),spatial_size=CFG.img_size)])

    img_paths = df[CFG.img_paths]
    if CFG.fcm_mask=="all":
        csf_mask_paths = [p.replace("images_scaled_voxel_spacing", f"fcm_norm_csf")[:-7] + f"csf_membership.nii.gz" for p in img_paths]
        gm_mask_paths = [p.replace("images_scaled_voxel_spacing", f"fcm_norm_gm")[:-7] + f"gm_membership.nii.gz" for p in img_paths]
        wm_mask_paths = [p.replace("images_scaled_voxel_spacing", f"fcm_norm_wm")[:-7] + f"wm_membership.nii.gz" for p in img_paths]
    else:
        csf_mask_paths = [p.replace("images_scaled_voxel_spacing", f"fcm_norm_{CFG.fcm_mask}")[:-7] + f"{CFG.fcm_mask}_membership.nii.gz" for p in img_paths]
        
    n_samples = len(df['id'])*int((np.ceil(256/CFG.slice_sampling_interval)))
    shared_array_base = mp.Array(ctypes.c_bool, n_samples*img_size*img_size)
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(n_samples,img_size,img_size)

    curr_idx = 0
    for idx in progress_bar(range(len(df)),len(df)):
        if CFG.fcm_mask=="all":
            csf_mask_fn = csf_mask_paths[idx]
            gm_mask_fn = gm_mask_paths[idx]
            wm_mask_fn = wm_mask_paths[idx]
            csf_mask = np.array(nib.load(f"{root_dir}/{csf_mask_fn}").get_fdata() > 0.5,dtype=np.bool_)
            gm_mask = np.array(nib.load(f"{root_dir}/{gm_mask_fn}").get_fdata() > 0.5,dtype=np.bool_)
            wm_mask = np.array(nib.load(f"{root_dir}/{wm_mask_fn}").get_fdata() > 0.5,dtype=np.bool_)
            csf_mask = csf_mask & gm_mask & wm_mask
        else:
            csf_mask_fn = csf_mask_paths[idx]
            csf_mask = np.array(nib.load(f"{root_dir}/{csf_mask_fn}").get_fdata() > 0.5,dtype=np.bool_)
        #change from sideways orientation to top down or cfg specified
        csf_mask = np.transpose(csf_mask,CFG.transpose)[None,:,:,:]
        tfms = monai_T({'image':csf_mask})
        csf_mask = tfms['image'].squeeze()
        
        #slice subsampling
        csf_mask = csf_mask[::CFG.slice_sampling_interval]
        
        c = csf_mask.shape[0]
        shared_array[curr_idx:curr_idx+c] = csf_mask
        curr_idx += c
    return shared_array

def create_3d_shared_arrays(CFG,df,root_dir):
    img_size = CFG.img_size[0]
    monai_T = T.Compose([T.CenterSpatialCropd(keys=("image","seg"),roi_size=CFG.img_size),
                         T.SpatialPadd(keys=("image","seg"),spatial_size=CFG.img_size)])
    img_paths = df[CFG.img_paths]
    segm_paths = df[CFG.segm_paths]
        
    n_samples = len(df['id'])
    shared_array_base = mp.Array(ctypes.c_float, n_samples*img_size*img_size*img_size)
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(n_samples,img_size,img_size,img_size)

    shared_array_base2 = mp.Array(ctypes.c_float, n_samples*img_size*img_size*img_size)
    shared_array2 = np.ctypeslib.as_array(shared_array_base2.get_obj())
    shared_array2 = shared_array2.reshape(n_samples, img_size,img_size,img_size)            

    curr_idx = 0
    for idx in progress_bar(range(len(df)),len(df)):
        img_fn = img_paths[idx]
        sgm_fn = segm_paths[idx]
        img = nib.load(f"{root_dir}/{img_fn}").get_fdata()
        segm = nib.load(f"{root_dir}/{sgm_fn}").get_fdata()  
        #img, segm = scale_voxel_spacing(idx, img, segm, df)
        if CFG.scale_mri:
            img = scale_mri(img, CFG.percentile)

        #change from sideways orientation to top down
        img = np.transpose(img,CFG.transpose)[None,:,:,:]
        segm = np.transpose(segm,CFG.transpose)[None,:,:,:]

        tfmed = monai_T({'image':img, 'seg':segm})       
        img = tfmed['image']
        segm = tfmed['seg']

        c = img.shape[0]
        shared_array[curr_idx:curr_idx+c] = img
        shared_array2[curr_idx:curr_idx+c] = segm
        curr_idx += c
    return shared_array,shared_array2