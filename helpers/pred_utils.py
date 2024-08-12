# Works for both pipeline and extra_layer

import numpy as np
import os
import numpy as np
import torch
from helpers.model import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from helpers.utils import *
import sunpy.io

from helpers.encoder import *




def collect_timestamp(
    srcBase: str,
    srcFlowBase: str,
    c_year: str,
    DEBUG_MODE = False,
    QUIET_MODE = False,
    ZARR_VERIFY = True
    ):  
    if QUIET_MODE == False:
        print(f'Loading data for year: {c_year}')
        print()

    # list of name of all folder that starts with c_year
    #Base_folder_list = [foldername for foldername in sorted(os.listdir(srcBase)) if
    #    foldername.startswith(c_year)]
    # !!! Folders in base and flowbase has the same name
    
    FlowBase_folder_list = [foldername for foldername in sorted(os.listdir(srcFlowBase)) if
        foldername.startswith(c_year)]
    
    foldername = None
    
    timestamp_list = []
    # Iterate through all and save to ZARR
    for i_main in range(len(FlowBase_folder_list)):
        foldername = FlowBase_folder_list[i_main]
        # use src to access file in folder for each measurement
        # src: srcBase and flowBase_folder list
        src = os.path.join(srcBase, foldername)

        print(f'src: {src}')
        
        for obsDate in sorted(os.listdir(src)):
            if obsDate.find("_") == -1 or not os.path.isdir(os.path.join(src,obsDate)):
                continue
            volFile = os.path.join(src, obsDate, obsDate+"_hmiObs.npz")
            #volInvFile = os.path.join(src, obsDate, obsDate+"_hmiInv.npz")
            volSPInvFile = os.path.join(srcFlowBase, foldername, obsDate, obsDate+"_flowedSPInv.npz")
            #volSPDisambigFile = os.path.join(srcFlowBase, foldername, obsDate, obsDate+"_flowedSPDisambig.npz")
            
            
            # check if file exists, if not, ignore this folder
            if os.path.exists(volFile) == False:
                continue
            
            if os.path.exists(volSPInvFile) == False:
                continue
            timestamp_list.append(obsDate)
    
    return np.array(timestamp_list)


def pred_pipeline(
    data_loader,
    model, 
    GLOBAL_DEVICE, 
    NPY_SAVE_PATH, 
    y_NORM_PARAM_DIR,
    TIMESTAMP_DIR,
    ):
    ########
    # Note:
    #
    # data_loader is the version without y
    #   
    ########
    with torch.no_grad():
        if not os.path.exists(NPY_SAVE_PATH):
            os.makedirs(NPY_SAVE_PATH)
        
        # Restore this after /Pool is online
        timestamp = np.load(TIMESTAMP_DIR)
        model = model.eval()
        pbar = tqdm(data_loader, ncols=300)
        

        last_i = None
        
        for i, batch in enumerate(pbar):
            # X: H,W,C
            # y: H,W     with correct shape
            X = batch['X'][0].to(GLOBAL_DEVICE)
            dmask = batch['disk_mask'][0].to(GLOBAL_DEVICE)
            
            # For X, remove all nans and apply disk mask
            X = X.masked_fill_(torch.isnan(X), 0)
            X = X.masked_fill_(~dmask, 0)
            X = X.float()
            
            pred_PATH = os.path.join(NPY_SAVE_PATH, f'{i}_predict_{timestamp[i][0]}.npy')
            pred = model(X)                       
            pred = pred.cpu()
                        
            
            y_norm_param = torch.from_numpy(np.load(y_NORM_PARAM_DIR).astype(np.float64))
            y_mean = y_norm_param[0,:]
            y_std = y_norm_param[1, :]
            
            pred = pred*y_std + y_mean
            np.save(pred_PATH, pred)
            
            last_i = i
            pbar.set_description(
                'itr {:<6}'
            .format(
                    i )) # batch time
        
        assert(last_i+1 == len(timestamp))
        timestamp_PATH = os.path.join(NPY_SAVE_PATH, 'TIMESTAMP.npy')
        np.save(timestamp_PATH, timestamp)



def inference_extra_layer_generate_X(X, fits_DIR):

    fits = sunpy.io.read_file(fits_DIR)
    OBS_VN = fits[1][1]['OBS_VN']
    OBS_VR = fits[1][1]['OBS_VR']
    OBS_VW = fits[1][1]['OBS_VW']
    N,C,H,W = X.shape
    
    ret = np.zeros((N,C+3,H,W))
    ret[:,:C,:,:] = X
    ret[:,C,:,:] = OBS_VN
    ret[:,C+1,:,:] = OBS_VR
    ret[:,C+2,:,:] = OBS_VW

    return ret

# Requires: npz file DIR, model, device
def inference_sample_full_disk(
    DIR:str, 
    NPZ_file_name:str,
    model, 
    X_norm_arr, 
    y_norm_arr, 
    GLOBAL_DEVICE, 
    mode='None', 
    fits_DIR = None,
    ):
    # X is in H,W,C
    npz = np.load(DIR)
    X = (npz["IQUV"]).astype(np.float64)
    mask = npz["mask"]


    if mode == 'extra_layer':
        X = inference_extra_layer_generate_X(X, fits_DIR, NPZ_file_name)
    
    X_mean = X_norm_arr[0,:]
    X_std = X_norm_arr[1,:]
    y_mean = y_norm_arr[0,:]
    y_std = y_norm_arr[1,:]
    
    X = (X-X_mean)/X_std
    X = torch.from_numpy(X).float().to(GLOBAL_DEVICE)
    mask = torch.from_numpy(mask).to(GLOBAL_DEVICE)
    
    X = torch.unsqueeze(X.permute(2,0,1), 0)
    mask = torch.unsqueeze(mask, 0)
    mask = torch.unsqueeze(mask, 0)
    
    with torch.no_grad():
        X = X.masked_fill_(~mask, 0)
        X = X.masked_fill_(~mask, 0)
        X = X.float()

        pred = model(X)
        pred = pred.cpu()
        pred = pred*y_std + y_mean
        return pred


# Use X_norm, Y_norm from training data in main
def inference_full_disk(
    DATA_DIR:str, 
    SAVE_PATH:str, 
    model,
    X_norm_DIR:str,
    y_norm_DIR:str,
    GLOBAL_DEVICE:str,
    mode = 'None',
    fits_DIR = None, 
    ):
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    npz_list = [filename for filename in sorted(os.listdir(DATA_DIR)) if filename.endswith("npz")]
    X_norm_arr = np.load(X_norm_DIR).astype(np.float64)
    y_norm_arr = np.load(y_norm_DIR).astype(np.float64)
    for npz_file_name in npz_list:
        pred = inference_sample_full_disk(os.path.join(DATA_DIR,npz_file_name), npz_file_name, model,
            X_norm_arr, y_norm_arr, GLOBAL_DEVICE, mode, fits_DIR)
        TMP_SAVE_PATH = os.path.join(SAVE_PATH, npz_file_name.replace(".npz","_predict.npy"))
        print(TMP_SAVE_PATH)
        np.save(TMP_SAVE_PATH, pred)
        



def classification_predict(
    data_loader,
    model, 
    GLOBAL_DEVICE, 
    NPY_SAVE_PATH, 
    TIMESTAMP_DIR,
    bins):
    with torch.no_grad():
        if not os.path.exists(NPY_SAVE_PATH):
            os.makedirs(NPY_SAVE_PATH)
            
        timestamp = np.load(TIMESTAMP_DIR)
        model = model.eval()

        pbar = tqdm(data_loader, ncols=200)

        last_i = None
        
        for i, batch in enumerate(pbar):

            # X: N,C,H,W   all depadded
            # y: N,C,H,W     with correct shape
            # dmask: N,C,H,W   0 is off-disk
            X = batch['X'][0].float().to(GLOBAL_DEVICE)
            dmask = batch['disk_mask'][0].to(GLOBAL_DEVICE)
            
            
            # For X, remove all nans and apply disk mask
            X = X.masked_fill_(torch.isnan(X), 0)
            X = X.masked_fill_(~dmask, 0)
            X = X.float()
            
            pred_PATH = os.path.join(NPY_SAVE_PATH, f'{i}_predict_{timestamp[i][0]}.npy')
            pred = model(X)
            pred = pred.cpu()
            
            
            pred = decoder_most_likely(pred, bins)
            print('Most Likely')
            #pred = decoder_median(pred, bins)

            np.save(pred_PATH, pred)
            last_i = i
            pbar.set_description(
                'itr {:<6}'
            .format(
                    i )) # batch time
            

        assert(last_i+1 == len(timestamp))
        
        timestamp_PATH = os.path.join(NPY_SAVE_PATH, 'TIMESTAMP.npy')
        np.save(timestamp_PATH, timestamp)




def classification_predict_with_noise(
    data_loader,
    model, 
    GLOBAL_DEVICE, 
    NPY_SAVE_PATH, 
    TIMESTAMP_DIR,
    bins,
    whether_CI = False):
    with torch.no_grad():
        if not os.path.exists(NPY_SAVE_PATH):
            os.makedirs(NPY_SAVE_PATH)
            
        timestamp = np.load(TIMESTAMP_DIR)
        model = model.eval()

        pbar = tqdm(data_loader, ncols=200)
        last_i = None
        
        LW_SAVE_PATH = None 
        HW_SAVE_PATH = None 
        if whether_CI == True:
            LW_SAVE_PATH = os.path.join(NPY_SAVE_PATH, 'Lower_CI')
            os.makedirs(LW_SAVE_PATH)
            HW_SAVE_PATH = os.path.join(NPY_SAVE_PATH, 'Higher_CI')
            os.makedirs(HW_SAVE_PATH)
            NPY_SAVE_PATH = os.path.join(NPY_SAVE_PATH, 'Prediction')
            os.makedirs(NPY_SAVE_PATH)
        
        for i, batch in enumerate(pbar):

            # X: N,C,H,W   all depadded
            # y: N,C,H,W     with correct shape
            # dmask: N,C,H,W   0 is off-disk
            X = batch['X'][0].float().to(GLOBAL_DEVICE)            
            dmask = batch['disk_mask'][0].to(GLOBAL_DEVICE)
            

            # For X, remove all nans and apply disk mask
            X = X.masked_fill_(torch.isnan(X), 0)
            X = X.masked_fill_(~dmask, 0)
            X = X.float()
            
            pred_PATH = os.path.join(NPY_SAVE_PATH, f'{i}_predict_{timestamp[i][0]}.npy')
            pred = model(X)
            pred = pred.cpu()
            
            
            if 'spInv_Field_Azimuth' in NPY_SAVE_PATH or 'spDisambig_Field_Azimuth_Disamb' in NPY_SAVE_PATH:
                print('With Noise Azimuth')
                if whether_CI == True:
                    pred, lw, hw = decoder_median_with_noise_Azimuth(pred, bins, whether_CI)
                    lw_PATH = os.path.join(LW_SAVE_PATH,  f'{i}_lb_{timestamp[i][0]}.npy')
                    hw_PATH = os.path.join(HW_SAVE_PATH,  f'{i}_hb_{timestamp[i][0]}.npy')
                    np.save(lw_PATH, lw)
                    np.save(hw_PATH, hw)
                else:
                    pred = decoder_median_with_noise_Azimuth(pred, bins, whether_CI)
                    
            else:
                print('With Noise base')
                if whether_CI == True:
                    pred , lw, hw = decoder_median_with_noise_base(pred, bins, whether_CI)
                    lw_PATH = os.path.join(LW_SAVE_PATH,  f'{i}_lb_{timestamp[i][0]}.npy')
                    hw_PATH = os.path.join(HW_SAVE_PATH,  f'{i}_hb_{timestamp[i][0]}.npy')
                    np.save(lw_PATH, lw)
                    np.save(hw_PATH, hw)
                else:
                    pred = decoder_median_with_noise_base(pred, bins, whether_CI)
                

            np.save(pred_PATH, pred)

            last_i = i
            pbar.set_description(
                'itr {:<6}'
            .format(
                    i )) # batch time
            

        assert(last_i+1 == len(timestamp))
        
        timestamp_PATH = os.path.join(NPY_SAVE_PATH, 'TIMESTAMP.npy')

        np.save(timestamp_PATH, timestamp)




# Requires: npz file DIR, model, device
def inference_sample_full_disk_classification(
    bins,
    DIR:str, 
    NPZ_file_name:str,
    model, 
    X_norm_arr,  
    GLOBAL_DEVICE, 
    mode='None', 
    fits_DIR = None,
    ):
    
    # X is in H,W,C
    npz = np.load(DIR)
    X = (npz["IQUV"]).astype(np.float64)
    
    mask = npz["mask"]


    if mode == 'extra_layer':
        quit()
        X = inference_extra_layer_generate_X(X, fits_DIR, NPZ_file_name)
    
    X_mean = X_norm_arr[0,:]
    X_std = X_norm_arr[1,:]
    
    
    X = (X-X_mean)/X_std
    X = torch.from_numpy(X).float().to(GLOBAL_DEVICE)
    mask = torch.from_numpy(mask).to(GLOBAL_DEVICE)
    
    X = torch.unsqueeze(X.permute(2,0,1), 0)
    mask = torch.unsqueeze(mask, 0)
    mask = torch.unsqueeze(mask, 0)
    
    with torch.no_grad():
        X = X.masked_fill_(~mask, 0)
        X = X.masked_fill_(~mask, 0)
        X = X.float()

        pred = model(X)
        pred = pred.cpu()
        #pred = decoder_most_likely(pred, bins)
        pred = decoder_median(pred, bins)
        return pred


# Use X_norm, Y_norm from training data in main
def inference_full_disk_classification(
    DATA_DIR:str, 
    SAVE_PATH:str, 
    model,
    X_norm_DIR:str,
    bins_DIR:str,
    GLOBAL_DEVICE:str,
    mode = 'None',
    fits_DIR = None, 
    ):
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
        
    bins = np.load(bins_DIR)
    bins = torch.from_numpy(bins)
    npz_list = [filename for filename in sorted(os.listdir(DATA_DIR)) if filename.endswith("npz")]
    X_norm_arr = np.load(X_norm_DIR).astype(np.float64)
    
    for npz_file_name in npz_list:
        pred = inference_sample_full_disk_classification(bins, os.path.join(DATA_DIR,npz_file_name), npz_file_name, model,
            X_norm_arr, GLOBAL_DEVICE, mode, fits_DIR)
        TMP_SAVE_PATH = os.path.join(SAVE_PATH, npz_file_name.replace(".npz","_predict.npy"))
        print(TMP_SAVE_PATH)
        np.save(TMP_SAVE_PATH, pred)
        

