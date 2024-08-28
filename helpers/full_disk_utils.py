import numpy as np
import os, json, sys
import numpy as np
import torch
import shutil
import subprocess
from helpers.model import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from helpers.utils import *
import astropy.io.fits as fits 
import astropy.io.fits
import sunpy.map
import astropy.units as u
import matplotlib.pyplot as plt
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

def getMetaData(HMIMap):
    H, W = HMIMap.data.shape[0], HMIMap.data.shape[1]
    HMIX, HMIY = np.meshgrid(np.array(range(W)), np.array(range(H)))
    sc = HMIMap.pixel_to_world(HMIX*u.pix, HMIY*u.pix)
    tx = sc.Tx.arcsec
    ty = sc.Ty.arcsec
    txs = sc.heliographic_stonyhurst.lon.deg
    tys = sc.heliographic_stonyhurst.lat.deg


    #We need to flip tx and ty since the transformation applies to the 180-adjusted
    #coordinates
    tx = tx[::-1,::-1]; ty = ty[::-1,::-1]
    txs, tys = txs[::-1,::-1], tys[::-1,::-1]
    mask = ~np.isnan(txs)

    return np.dstack([tx[:,:,None], ty[:,:,None], mask[:,:,None]]), np.dstack([txs[:,:,None], tys[:,:,None]]), mask

'''
# target: Save directory
# file_name: input fits file name
# headerSrc: fits header
# !!! Takes one rotation for incoming data (combing fits file already rotate once)
def pack_to_fits(target, file_name, imageData, headerSrc, y_name, partition, compress=True, whether_flip = True):
    short_name = y_name.split('_', 1)[1]
    parts = file_name.split(".")
    short_name_partition = short_name + partition
    parts[-3:] = [f'{short_name_partition}', f'fits']
    file_name = ".".join(parts)
    file_name = file_name.replace("S_720s", 'SuperSynthIA')
    save_DIR = os.path.join(target, file_name)
    # Flip the data
    if partition == '_orig_logit':
        imageData = imageData[:, ::-1, ::-1]
    elif y_name == '_mask' or whether_flip == False:
        imageData = imageData
    elif y_name == 'spDisambig_Bt' and partition != '_err':
        imageData = -imageData[::-1,::-1]
    else:
        imageData = imageData[::-1,::-1]
    header0 = (headerSrc[0].header).copy()

    data = [fits.PrimaryHDU(data=None, header=None)]
    if compress:
        data += [fits.CompImageHDU(data=imageData, header=header0, compression_type='RICE_1', tile_shape=(64, 64), quantize_level=-0.01)]
    else:
        data += [fits.ImageHDU(data=imageData, header=header0)]

    
    hdul = fits.HDUList(data)
    hdul.writeto(save_DIR, overwrite=True)

'''

def pack_to_fits(target, file_name, imageData, headerSrc, y_name, partition, compress=True, whether_flip=True):
    short_name = y_name.split('_', 1)[1]
    parts = file_name.split(".")
    short_name_partition = short_name + partition
    parts[-3:] = [f'{short_name_partition}', f'fits']
    file_name = ".".join(parts)
    file_name = file_name.replace("S_720s", 'SuperSynthIA')
    save_DIR = os.path.join(target, file_name)
    
    # Flip the data
    if partition == '_orig_logit':
        imageData = imageData[:, ::-1, ::-1]
    elif y_name == '_mask' or whether_flip == False:
        imageData = imageData
    elif y_name == 'spDisambig_Bt' and partition != '_err':
        imageData = -imageData[::-1, ::-1]
    else:
        imageData = imageData[::-1, ::-1]
    
    header0 = (headerSrc[0].header).copy()
    
    data = [fits.PrimaryHDU(data=None, header=None)]
    data += [fits.ImageHDU(data=imageData, header=header0)]
    
    hdul = fits.HDUList(data)
    uncompressed_save_DIR = save_DIR.replace('.fits', '_uncompressed.fits')
    hdul.writeto(uncompressed_save_DIR, overwrite=True)
    
    if compress:
        # Check if fpack is available
        if shutil.which('fpack') is not None:
            # Use fpack with RICE compression to compress the FITS file
            t_command = " ".join(['fpack', '-O', save_DIR, '-q', '-0.01', uncompressed_save_DIR])
            os.system(t_command)
            # Remove the uncompressed file after compression
            os.remove(uncompressed_save_DIR)
        else:
            print(f'fpack not found. Saving uncompressed file: {uncompressed_save_DIR}')
    else:
        # Rename the uncompressed file to the final name
        os.rename(uncompressed_save_DIR, save_DIR)


# For 80 days of data


from helpers.pred_utils import *

def nnInterpNaN(X):
    """Given HxWxC image X, nearest neighbor interpolate all pixels with a nan
        in any channel. Not very efficient"""
    M = np.any(np.isnan(X),axis=2)

    distanceIndMulti = scipy.ndimage.distance_transform_edt(M, return_distances=False, return_indices=True)
    distanceInd = np.ravel_multi_index(distanceIndMulti, M.shape)

    X2 = X.copy()
    for c in range(X.shape[2]):
        Xc = X[:,:,c]
        X2[:,:,c] = Xc.ravel()[distanceInd]
    return X2 

# Require: DATA_DIR: /nfs/turbo/.../hmi.s_720s.timestamp_TAI.1.XX.fits
# Take one 180 rotation
def concat_full_disk(DATA_DIR:str, filename):
    foldername = os.path.join(DATA_DIR, filename)
    I0_DIR = foldername.replace("XX", "I0")
    I0_arr = (astropy.io.fits.open(I0_DIR)[1].data)[::-1, ::-1]
    hmi_Map = sunpy.map.Map(I0_DIR)
    trash1, trash2, mask = getMetaData(hmi_Map)
    I0_DIR = None
    
    
    mask = np.expand_dims(mask, axis=0)
    mask = np.expand_dims(mask, axis=0)
    # X is in N,C,H,W
    X = np.zeros((1, 24, I0_arr.shape[0], I0_arr.shape[1]))
    X[:,0,:,:] = I0_arr
    
    
    TIMESTAMP = filename[11:26]
    layer = 0
    # For I
    for i in range(6):
        TMP_I_DIR = foldername.replace("XX", f"I{i}")
        arr = (astropy.io.fits.open(TMP_I_DIR)[1].data)[::-1, ::-1]
        X[:,layer,:,:] = arr
        layer += 1
    
    # For Q
    for i in range(6):
        TMP_I_DIR = foldername.replace("XX", f"Q{i}")
        arr = (astropy.io.fits.open(TMP_I_DIR)[1].data)[::-1, ::-1]
        X[:,layer,:,:] = arr
        layer += 1
    
    # For U
    for i in range(6):
        TMP_I_DIR = foldername.replace("XX", f"U{i}")
        arr = (astropy.io.fits.open(TMP_I_DIR)[1].data)[::-1, ::-1]
        X[:,layer,:,:] = arr
        layer += 1
    
    # For V
    for i in range(6):
        TMP_I_DIR = foldername.replace("XX", f"V{i}")
        arr = (astropy.io.fits.open(TMP_I_DIR)[1].data)[::-1, ::-1]
        X[:,layer,:,:] = arr
        layer += 1
    
    return X, mask, TIMESTAMP 
    

# Requires: npz file DIR, model, device
def inference_sample_full_disk_80_days(
    DIR:str, 
    fits_file_name:str,
    model, 
    X_norm_arr, 
    y_norm_arr, 
    GLOBAL_DEVICE, 
    is_classification,
    mode='None', 
    bins = None,
    y_name = None,
    save_std = False,
    save_CI = False,
    save_orig_logit = False,
    reproducible = False,
    fixed_noise = None,
    ):
    
    # X is in H,W,C
    X, mask, TIMESTAMP = concat_full_disk(DIR, fits_file_name)


    if mode == 'extra_layer':
        tmp_fits_dir = os.path.join(DIR, fits_file_name)
        tmp_I0_DIR = tmp_fits_dir.replace("XX", "I0")
        X = inference_extra_layer_generate_X(X, tmp_I0_DIR)
    
    X_mean = X_norm_arr[0,:]
    X_std = X_norm_arr[1,:]
    y_mean = y_norm_arr[0,:]
    y_std = y_norm_arr[1,:]
    
    X = torch.from_numpy(X).float().to(GLOBAL_DEVICE)
    mask = torch.from_numpy(mask).to(GLOBAL_DEVICE)
    # X: 1,24,4096,4096
    X = X.squeeze()
    X = X.permute(1,2,0)
    # X: 4096, 4096, 24
    
    #X = (X-X_mean)/X_std
    X -= X_mean
    X /= X_std
    
    
    X = torch.unsqueeze(X.permute(2,0,1), 0)
    
    # X: 1,24,4096,4096
    # mask: 1,24,4096,4096
    TIMESTAMP = extract_timestamp_from_fits(fits_file_name)
    
    
    with torch.no_grad():
        X = X.masked_fill_(~mask, 0)
        X = X.float()
        # X: 1,24,4096,4096
        
        # If X contain nan, apply nnInterp
        if torch.any(torch.isnan(X)):
            X = X.squeeze()
            X = X.permute(1,2,0)
            # X: 4096,4096, 24
            
            X = X.cpu().numpy()
            X = nnInterpNaN(X)
            X = torch.from_numpy(X).float().to(GLOBAL_DEVICE)
            X = torch.unsqueeze(X.permute(2,0,1), 0)
            assert(torch.any(torch.isnan(X)) == False)
            print(f'HAVE NAN: {fits_file_name}')
            
        
        # Separate X into chunks
        # Turn down if your GPU memory size is insufficient
        # Make Sure that NUM_CHUNKS * CHUNK_SIZE = 4096
        CHUNK_SIZE = 256
        #CHUNK_SIZE = 1024
        
        if CHUNK_SIZE % 64 != 0:
            print("Error: CHUNK_SIZE must be a multiple of 64")
            sys.exit(1)
        
        NUM_CHUNKS = 4096 // CHUNK_SIZE
        if CHUNK_SIZE * NUM_CHUNKS != 4096:
            print("Error: CHUNK_SIZE * NUM_CHUNKS must equal 4096")
            sys.exit(1)
        
        FINE_CHUNKS = NUM_CHUNKS+(NUM_CHUNKS-1)
        FINE_OFFSET = CHUNK_SIZE //2

        Yp = []
        yp_std = []
        yp_low_CI = []
        yp_high_CI = []
        yp_orig_prob = []
        for y in range(FINE_CHUNKS):
            Yp.append([])
            yp_std.append([])
            yp_low_CI.append([])
            yp_high_CI.append([])
            yp_orig_prob.append([])
            for x in range(FINE_CHUNKS):
                #savePath = lambda o,x: np.savez_compressed(os.path.join(targetFolders[o],date+("_%d%d.npz" % (y,x))),X=x)
                sy = y*FINE_OFFSET 
                sx = x*FINE_OFFSET
                ey = sy + CHUNK_SIZE
                ex = sx + CHUNK_SIZE
                X_chunk = X[:, :, sy:ey, sx:ex]
                #print(X_chunk.shape)
                pred_chunk = None 
                low_CI, high_CI, pred_std, orig_prob = None, None, None, None
                
                # Build the hashing key for reproducibility
                hashing_key = None
                if reproducible == True:
                    hashing_key = (TIMESTAMP, sy, sx)
                
                if is_classification == False:
                    assert(False)
                    pred_chunk = model(X_chunk)
                    pred_chunk = pred_chunk*y_std + y_mean
                    pred_chunk = pred_chunk.squeeze().cpu()
                else:
                    pred_chunk = model(X_chunk)
                    if save_orig_logit == True:
                        orig_prob = pred_chunk.clone().detach().squeeze().cpu().numpy()
                    if 'spInv_Field_Azimuth' == y_name or 'spDisambig_Field_Azimuth_Disamb' == y_name:
                        pred_chunk, low_CI, high_CI, pred_std = decoder_median_with_noise_Azimuth(pred_chunk, bins, whether_CI = save_CI, whether_std = save_std, 
                                                                                                  reproducible = reproducible, fixed_noise = fixed_noise, hashing_key = hashing_key)
                    else:
                        pred_chunk, low_CI, high_CI, pred_std = decoder_median_with_noise_base(pred_chunk, bins, whether_CI = save_CI, whether_std = save_std, 
                                                                                               reproducible = reproducible, fixed_noise = fixed_noise, hashing_key = hashing_key)
                        
                    pred_chunk = pred_chunk.squeeze().cpu()
                    if save_std == True:
                        pred_std = pred_std.squeeze().cpu()
                    if save_CI == True:
                        low_CI = low_CI.squeeze().cpu()
                        high_CI = high_CI.squeeze().cpu()
                    
                
                x_range, y_range = None, None
                if y == 0:
                    y_range = [0, int(0.75*CHUNK_SIZE)]
                elif y == FINE_CHUNKS-1:
                    y_range = [int(0.25*CHUNK_SIZE), CHUNK_SIZE]
                else: 
                    y_range = [int(0.25*CHUNK_SIZE), int(0.75*CHUNK_SIZE)]

                if x == 0:
                    x_range = [0, int(0.75*CHUNK_SIZE)]
                elif x == FINE_CHUNKS-1:
                    x_range = [int(0.25*CHUNK_SIZE), CHUNK_SIZE]
                else: 
                    x_range = [int(0.25*CHUNK_SIZE), int(0.75*CHUNK_SIZE)]
            

                wanted_chunk = pred_chunk[y_range[0]:y_range[1], x_range[0]:x_range[1]]
                Yp[-1].append(wanted_chunk)
                
                if save_std == True:
                    yp_std[-1].append(pred_std[y_range[0]:y_range[1], x_range[0]:x_range[1]])
                if save_CI == True:
                    yp_low_CI[-1].append(low_CI[y_range[0]:y_range[1], x_range[0]:x_range[1]])
                    yp_high_CI[-1].append(high_CI[y_range[0]:y_range[1], x_range[0]:x_range[1]])
                if save_orig_logit == True:
                    # (c, H, W)
                    yp_orig_prob[-1].append(orig_prob[:, y_range[0]:y_range[1], x_range[0]:x_range[1]])
            
            Yp[-1] = np.hstack(Yp[-1])
            
            if save_std == True:
                yp_std[-1] = np.hstack(yp_std[-1])
            if save_CI == True:
                yp_low_CI[-1] = np.hstack(yp_low_CI[-1])
                yp_high_CI[-1] = np.hstack(yp_high_CI[-1])
            if save_orig_logit == True:
                yp_orig_prob[-1] = np.concatenate(yp_orig_prob[-1], axis=2)
        Yp = np.vstack(Yp)
        Yp = Yp.astype(np.float32)
        if save_std == True:
            yp_std = (np.vstack(yp_std)).astype(np.float32)
        if save_CI == True:
            yp_low_CI = (np.vstack(yp_low_CI)).astype(np.float32)
            yp_high_CI = (np.vstack(yp_high_CI)).astype(np.float32)
        if save_orig_logit == True:
            yp_orig_prob = (np.concatenate(yp_orig_prob, axis=1)).astype(np.float32)
        return Yp, yp_low_CI, yp_high_CI, yp_std, yp_orig_prob



def save_state(state, filename='checkpoint.json'):
    with open(filename, 'w') as f:
        json.dump(state, f)

def load_state(filename='checkpoint.json'):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None

def process_file(fits_file_name, DATA_DIR, SAVE_PATH, model, X_norm_arr, y_norm_arr, GLOBAL_DEVICE, 
                 is_CLASSIFICATION, mode, bins, y_name, save_std, save_CI, save_as_FITS, 
                 save_orig_logit, lock_file, reproducible, fixed_noise):
    try:
        pred, low_CI, high_CI, pred_std, orig_prob = inference_sample_full_disk_80_days(
            DATA_DIR, fits_file_name, model, X_norm_arr, y_norm_arr, GLOBAL_DEVICE, 
            is_CLASSIFICATION, mode, bins, y_name, save_std, save_CI, save_orig_logit, 
            reproducible, fixed_noise
        )
        
        utils_make_dir(SAVE_PATH)
        if not save_as_FITS:
            if y_name == 'spDisambig_Bt':
                pred = -pred[::-1, ::-1]
                low_CI = -low_CI[::-1, ::-1]
                high_CI = -high_CI[::-1, ::-1]
                pred_std = pred_std[::-1, ::-1]
                orig_prob = orig_prob[:, ::-1, ::-1]
            else:
                pred = pred[::-1, ::-1]
                low_CI = low_CI[::-1, ::-1]
                high_CI = high_CI[::-1, ::-1]
                pred_std = pred_std[::-1, ::-1]
                orig_prob = orig_prob[:, ::-1, ::-1]
            
            TMP_pred_SAVE_PATH = os.path.join(SAVE_PATH, fits_file_name.replace("XX.fits", "_predict.npy"))
            np.save(TMP_pred_SAVE_PATH, pred)
            if save_CI:
                TMP_low_CI_SAVE_PATH = os.path.join(SAVE_PATH, fits_file_name.replace("XX.fits", "_low_CI.npy"))
                TMP_high_CI_SAVE_PATH = os.path.join(SAVE_PATH, fits_file_name.replace("XX.fits", "_high_CI.npy"))
                np.save(TMP_low_CI_SAVE_PATH, low_CI)
                np.save(TMP_high_CI_SAVE_PATH, high_CI)
            if save_std:
                TMP_pred_std_SAVE_PATH = os.path.join(SAVE_PATH, fits_file_name.replace("XX.fits", "_predict_std.npy"))
                np.save(TMP_pred_std_SAVE_PATH, pred_std)
            if save_orig_logit:
                TMP_orig_prob_SAVE_PATH = os.path.join(SAVE_PATH, fits_file_name.replace("XX.fits", "_orig_prob.npy"))
                np.save(TMP_orig_prob_SAVE_PATH, orig_prob)
        else:
            pack_to_fits(SAVE_PATH, fits_file_name, pred, fits.open(os.path.join(DATA_DIR, fits_file_name.replace("XX.fits", "I0.fits"))), y_name, '') 
            if save_CI:
                pack_to_fits(SAVE_PATH, fits_file_name, low_CI, fits.open(os.path.join(DATA_DIR, fits_file_name.replace("XX.fits", "I0.fits"))), y_name, '_low_CI') 
                pack_to_fits(SAVE_PATH, fits_file_name, high_CI, fits.open(os.path.join(DATA_DIR, fits_file_name.replace("XX.fits", "I0.fits"))), y_name, '_high_CI')
            if save_std:
                pack_to_fits(SAVE_PATH, fits_file_name, pred_std, fits.open(os.path.join(DATA_DIR, fits_file_name.replace("XX.fits", "I0.fits"))), y_name, '_err')
            if save_orig_logit:
                pack_to_fits(SAVE_PATH, fits_file_name, orig_prob, fits.open(os.path.join(DATA_DIR, fits_file_name.replace("XX.fits", "I0.fits"))), y_name, '_orig_logit', compress=False)
        
        # Update lock file after successful processing
        with open(lock_file, 'a') as f:
            f.write(f"{fits_file_name}\n")
    except Exception as e:
        print(f"Error processing file {fits_file_name}: {e}")

def inference_full_disk_80_days(
    DATA_DIR:str, 
    SAVE_PATH:str, 
    model,
    X_norm_DIR:str,
    y_norm_DIR:str,
    GLOBAL_DEVICE:str,
    mode = 'None',
    is_CLASSIFICATION = False, 
    OBS_norm_dir_list = [],
    bins = None,
    y_name = None,
    save_std = False,
    save_CI = False,
    save_as_FITS = False,
    save_orig_logit = False,
    parallel = True, 
    parallel_workers = 4,
    reproducible = False,
    ):
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    
    file_list = [filename.replace("I0", "XX") for filename in sorted(os.listdir(DATA_DIR)) if filename.endswith(".I0.fits")]
    X_norm_arr = np.load(X_norm_DIR).astype(np.float64)
    y_norm_arr = np.load(y_norm_DIR).astype(np.float64)
    
    if OBS_norm_dir_list:
        COMB_MEAN = np.zeros((24 + len(OBS_norm_dir_list)))
        COMB_STD = np.zeros_like(COMB_MEAN)
        COMB_MEAN[:24] = X_norm_arr[0, :]
        COMB_STD[:24] = X_norm_arr[1, :]
        for i, obs_norm_dir in enumerate(OBS_norm_dir_list):
            tmp_norm = np.load(obs_norm_dir)
            COMB_MEAN[24 + i] = tmp_norm[0, :]
            COMB_STD[24 + i] = tmp_norm[1, :]
        X_norm_arr = np.zeros((2, 24 + len(OBS_norm_dir_list)))
        X_norm_arr[0, :] = COMB_MEAN
        X_norm_arr[1, :] = COMB_STD
    
    X_norm_arr = torch.from_numpy(X_norm_arr).float().to(GLOBAL_DEVICE)
    y_norm_arr = torch.from_numpy(y_norm_arr).float().to(GLOBAL_DEVICE)
    
    # Define the lock file path within the SAVE_PATH directory
    lock_file = os.path.join(SAVE_PATH, 'generated_files.txt')
    
    # Load the processed files from the lock file
    processed_files = set()
    if os.path.exists(lock_file):
        with open(lock_file, 'r') as f:
            processed_files = set(f.read().splitlines())
            
    # If reproducible, load the saved noise
    fixed_noise = None
    if reproducible:
        fixed_noise = torch.load('./helpers/triangular_noise.pt')
        fixed_noise = fixed_noise.to(GLOBAL_DEVICE)

    if parallel:
        print(f'Running with parallelism with {parallel_workers} workers')
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = []
            for fits_file_name in file_list:
                if fits_file_name not in processed_files:
                    futures.append(executor.submit(process_file, fits_file_name, DATA_DIR, SAVE_PATH, model, X_norm_arr, y_norm_arr, 
                                                   GLOBAL_DEVICE, is_CLASSIFICATION, mode, bins, y_name, save_std, save_CI, save_as_FITS, 
                                                   save_orig_logit, lock_file, reproducible, fixed_noise))
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in future: {e}")
    else:
        print('Running without parallelism')
        for fits_file_name in file_list:
            if fits_file_name not in processed_files:
                process_file(fits_file_name, DATA_DIR, SAVE_PATH, model, X_norm_arr, y_norm_arr, GLOBAL_DEVICE, is_CLASSIFICATION,
                             mode, bins, y_name, save_std, save_CI, save_as_FITS, save_orig_logit, lock_file, reproducible, fixed_noise)


def replace_nans(
    DATA_DIR:str, 
    SAVE_PATH:str, 
    model,
    X_norm_DIR:str,
    y_norm_DIR:str,
    GLOBAL_DEVICE:str,
    mode = 'None',
    fits_DIR = None,
    PRED_DIR = None):
    file_list = sorted(os.listdir(PRED_DIR))
    X_norm_arr = np.load(X_norm_DIR).astype(np.float64)
    y_norm_arr = np.load(y_norm_DIR).astype(np.float64)
    X_norm_arr = torch.from_numpy(X_norm_arr).float().to(GLOBAL_DEVICE)
    y_norm_arr = torch.from_numpy(y_norm_arr).float().to(GLOBAL_DEVICE)
    for filename in file_list:
        pred_arr = np.load(os.path.join(PRED_DIR, filename))
        
        if np.any(np.isnan(pred_arr)):
            fits_file_name = filename.replace("_predict.npy", "XX.fits")
            pred = inference_sample_full_disk_80_days(DATA_DIR, fits_file_name, model,
                X_norm_arr, y_norm_arr, GLOBAL_DEVICE, mode, fits_DIR)
            TMP_SAVE_PATH = os.path.join(SAVE_PATH, fits_file_name.replace("XX.fits","_predict.npy"))
            print(TMP_SAVE_PATH)
            np.save(TMP_SAVE_PATH, pred)
    

# --------------------------------------------------------------------
# Handles unreliable points in the data


# Given a fits file, generate a uncertainty mask of the hole
def uncertainty_mask_generation(array):
    # Calculate the 99.9999th percentile
    p99 = np.percentile(array, 99.99)

    # Generate a mask for values greater than the percentile and greater than 1000
    mask = (array > p99) & (array > 300)

    return mask

def get_data_from_fits(fits_dir):
    arr = fits.open(fits_dir)[1].data
    return arr


def extract_date_time(file_name):
    parts = file_name.split('.')
    date_time_str = parts[2]
    # Exclude the _TAI
    date_time_str = date_time_str[:-4]
    return date_time_str

# Turn '20160513_204800' into dt
def date_time_to_dt(date_time_str):
    dt = datetime(year=int(date_time_str[:4]), month=int(date_time_str[4:6]), day=int(date_time_str[6:8]), hour=int(date_time_str[9:11]), minute=int(date_time_str[11:13]))
    return dt

# Given an array, calculate the average of the center 1024x1024
def calculate_I_center(arr):
    center_x, center_y = arr.shape[0] // 2, arr.shape[1] // 2

    # Define the start and end of the slice
    start_x, end_x = center_x - 512, center_x + 512
    start_y, end_y = center_y - 512, center_y + 512

    # Take the slice
    center_slice = arr[start_x:end_x, start_y:end_y]

    # Calculate the average
    average = center_slice.mean()
    return average
    

# Given a fits file directory, calcualte mu
def continuum_calculate_mu(fits_src, ):
    # Load it into a sunpy map
    HMIMap = sunpy.map.Map(fits_src)
    
    date_time_str = extract_date_time(os.path.basename(fits_src))
    dt = date_time_to_dt(date_time_str)

    # Create a meshgrid of the X / Y  
    H, W = HMIMap.data.shape[0], HMIMap.data.shape[1]
    HMIX, HMIY = np.meshgrid(np.array(range(W)), np.array(range(H)))

    # use sunpy to convert to coordinates -- the sc thing will give you
    # whatever coordinates you want. Don't do any of this manual, this is
    # just unbelievably painful
    sc = HMIMap.pixel_to_world(HMIX*u.pix, HMIY*u.pix) 

    '''
    # this is a cryptic expression that I found. I convinced myself it works
    # and it matches mu3 (the full formula), but I wanted to double check
    frac = sc.heliocentric.cylindrical.rho / sc.rsun.to(u.m)
    mu = (1-frac**2)**0.5
    '''

    # get lat/lon
    tx = sc.Tx.arcsec
    ty = sc.Ty.arcsec
    txs = sc.heliographic_stonyhurst.lon.deg
    tys = sc.heliographic_stonyhurst.lat.deg

    '''
    # fast one assuming b0 = 0
    mu2 = np.cos(np.radians(txs)) * np.cos(np.radians(tys))
    '''
    
    # full formula
    b0 = sunpy.coordinates.sun.B0(dt) # get b0
    b0 = b0.deg # strip the unit tag
    mu3 = np.sin(np.radians(b0))*np.sin(np.radians(tys)) + np.cos(np.radians(b0))*np.cos(np.radians(tys))*np.cos(np.radians(txs))
    return mu3

#   IQUV_DIGIT can be 1 or 3 in hmi.720S series
def is_valid_iquv_digit(base_file_name, digit, data_dir):
    I0_file_DIR = base_file_name.replace("XX", f"{digit}.I0")
    I5_file_DIR = base_file_name.replace("XX", f"{digit}.I5")
    I0_path = os.path.join(data_dir, I0_file_DIR)
    I5_path = os.path.join(data_dir, I5_file_DIR)
    return os.path.exists(I0_path) and os.path.exists(I5_path)


def continuum_process_file(file_name, Bp_PRED_DIR, Br_PRED_DIR, Bt_PRED_DIR, IQUV_DATA_DIR, MASK_SAVE_DIR, esti_u):
    Bp = get_data_from_fits(os.path.join(Bp_PRED_DIR, file_name.replace("XX", "Bp")))
    Br = get_data_from_fits(os.path.join(Br_PRED_DIR, file_name.replace("XX", "Br")))
    Bt = get_data_from_fits(os.path.join(Bt_PRED_DIR, file_name.replace("XX", "Bt")))
    B_mag = np.sqrt(Bp**2 + Br**2 + Bt**2)

    IQUV_base_file_name = file_name.replace('SuperSynthIA', "S_720s")
    IQUV_DIGIT = None
    # Test both possible values
    if is_valid_iquv_digit(IQUV_base_file_name, 1, IQUV_DATA_DIR):
        IQUV_DIGIT = 1
    elif is_valid_iquv_digit(IQUV_base_file_name, 3, IQUV_DATA_DIR):
        IQUV_DIGIT = 3
    else:
        raise ValueError("Neither 1 nor 3 is a valid IQUV_DIGIT")
    
    I0_file_DIR = IQUV_base_file_name.replace("XX", f"{IQUV_DIGIT}.I0")
    I5_file_DIR = IQUV_base_file_name.replace("XX", f"{IQUV_DIGIT}.I5")
    I0 = get_data_from_fits(os.path.join(IQUV_DATA_DIR, I0_file_DIR))
    I5 = get_data_from_fits(os.path.join(IQUV_DATA_DIR, I5_file_DIR))
    I_avg = (I0 + I5) / 2

    mu = continuum_calculate_mu(os.path.join(IQUV_DATA_DIR, I0_file_DIR))
    I_new = I_avg / (1.0 - esti_u*(1.0 - mu))

    proxy = 1.0 - (B_mag/np.nanmax(B_mag))
    mask = (proxy / I_new) / np.nanmedian(proxy / I_new) > 4
    mask = mask.astype(np.uint8)
    arr = (proxy / I_new) / np.nanmedian(proxy / I_new)

    #plt.imsave(os.path.join(MASK_SAVE_DIR, 'vis', file_name.replace(".fits", "_mask.png")), mask, cmap='gray')
    #plt.imsave(os.path.join(MASK_SAVE_DIR, 'vis', file_name.replace(".fits", "_br.png")), Br, vmin=-2000, vmax=2000)
    #plt.imsave(os.path.join(MASK_SAVE_DIR, 'vis', file_name.replace(".fits", "_ratio.png")), arr, vmin=0, vmax=4)
    pack_to_fits(MASK_SAVE_DIR, I0_file_DIR, mask, fits.open(os.path.join(IQUV_DATA_DIR, I0_file_DIR)), '_mask', '')


# KD's method of generating mask of uncertain pixles
def continuum_based_bad_point_identify(
    PRED_DATA_DIR,
    IQUV_DATA_DIR,
    esti_u=0.7,
    parallel=True,
    max_parallel_workers = 4
):
    Bp_PRED_DIR = os.path.join(PRED_DATA_DIR, 'spDisambig_Bp')
    Br_PRED_DIR = os.path.join(PRED_DATA_DIR, 'spDisambig_Br')
    Bt_PRED_DIR = os.path.join(PRED_DATA_DIR, 'spDisambig_Bt')
    MASK_SAVE_DIR = os.path.join(PRED_DATA_DIR, 'MASK')
    os.makedirs(MASK_SAVE_DIR, exist_ok=True)
    #os.makedirs(os.path.join(MASK_SAVE_DIR, 'vis'), exist_ok=True)
    file_list = [filename.replace("Bp", "XX") for filename in sorted(os.listdir(Bp_PRED_DIR)) if filename.endswith(".Bp.fits")]
    print(f'Processing {len(file_list)} files')
    if parallel:
        with ProcessPoolExecutor(max_workers = max_parallel_workers) as executor:
            executor.map(
                continuum_process_file,
                file_list,
                [Bp_PRED_DIR] * len(file_list),
                [Br_PRED_DIR] * len(file_list),
                [Bt_PRED_DIR] * len(file_list),
                [IQUV_DATA_DIR] * len(file_list),
                [MASK_SAVE_DIR] * len(file_list),
                [esti_u] * len(file_list)
            )
    else:
        for file_name in file_list:
            continuum_process_file(file_name, Bp_PRED_DIR, Br_PRED_DIR, Bt_PRED_DIR, IQUV_DATA_DIR, MASK_SAVE_DIR, esti_u)



from scipy.interpolate import RBFInterpolator


# Hole fixing Interpolation
def quickInterpolate(Z, invalidMask, interpolateFromDistance=6):
    """
    Do thin-plate spline fixing of bad pixels. 
    This makes a few assumptions to make it fast:
        a) it uses plane of the sky distance
        b) it only uses points near the holes for interpolation,
            which isn't strictly correct, but is good enough

    invalidMask is assumed to have the semantics of:
        { False: this pixel is valid aka should be kept
        { True: this pixel is invalid aka should be interpolated

    This will definitely go bad if Z contains NaNs
    """
    # invalidMask should be primarily 0 aka primarily not invalid 

    assert((Z.shape[0] == invalidMask.shape[0]) and
            (Z.shape[1] == invalidMask.shape[1]))

    ZReturn = Z.copy()

    # distance to nearest invalid pixel
    dt = scipy.ndimage.distance_transform_edt(~invalidMask)

    # get a grid
    Y, X = np.meshgrid(
            np.arange(Z.shape[0]).astype(float),
            np.arange(Z.shape[1]).astype(float),
            indexing='ij')

    # fill in places with 0 distance to invalid
    toFillIn = (dt == 0) 

    # fill in from places from (within the distance but not invalid)
    toFillFrom = (dt <= interpolateFromDistance) & (dt > 0)

    # create a matrix + vector for the locations + values
    evalFromLoc = np.hstack([
                     X[toFillFrom].reshape(-1,1), 
                     Y[toFillFrom].reshape(-1,1)
                    ])
    evalFromVal = Z[toFillFrom].reshape(-1,1)
   
    # set up an interpolator
    interpFunction = RBFInterpolator(
                        evalFromLoc, evalFromVal
                        )

    # interpolate
    toFillInValue = interpFunction(np.hstack([
                        X[toFillIn].reshape(-1,1), 
                        Y[toFillIn].reshape(-1,1)
                      ]))

    # re-fill
    ZReturn[toFillIn] = toFillInValue.reshape(-1)
    return ZReturn


# Given a fits file directory, produce interpoalted result based on mode
# MODE:
#   'Overwrite' : Overwrite the original fits file
#   'SaveNew' : Save fixed result to new directory
def Interpolation_packer(MODE, PRED_DATA_DIR, SAVE_DIR = None, parallel = True, max_parallel_workers = 4):
    print(f'Fixing {PRED_DATA_DIR}')
    VALID_MODES = ['Overwrite', 'SaveNew']
    assert MODE in VALID_MODES
    if MODE == 'SaveNew':
        assert SAVE_DIR != None
        os.makedirs(SAVE_DIR, exist_ok=True)
        
    Br_PRED_DIR = os.path.join(PRED_DATA_DIR, 'spDisambig_Br')
    MASK_SAVE_DIR = os.path.join(PRED_DATA_DIR, 'MASK')
    
    all_list = [filename.replace("Br", "XX") for filename in sorted(os.listdir(Br_PRED_DIR)) if filename.endswith(".Br.fits")]
    if parallel:
        # Use ProcessPoolExecutor to run tasks in parallel
        with ProcessPoolExecutor(max_workers= max_parallel_workers) as executor:
            executor.map(
                Interpolation_samp_parallel,
                [MODE] * len(all_list),
                [Br_PRED_DIR] * len(all_list),
                [MASK_SAVE_DIR] * len(all_list),
                all_list,
                [SAVE_DIR] * len(all_list)
            )
            
    else:
        for file_name in all_list:
            Interpolation_samp_parallel(MODE, Br_PRED_DIR, MASK_SAVE_DIR, file_name, SAVE_DIR)


def ssqrt(x):
    return np.sign(x) * np.sqrt(np.abs(x))

def file_name_to_pack_fits(file_name):
    file_name = file_name.replace('SuperSynthIA', "S_720s")
    file_name = file_name.replace("XX", "3.I0")
    return file_name

def Interpolation_samp_parallel(MODE, Br_PRED_DIR, MASK_SAVE_DIR, file_name, SAVE_DIR):
    log_file = os.path.join(Br_PRED_DIR, f"processed_files_{MODE}.log")
    
    # Check if the file has already been processed
    if os.path.exists(log_file):
        with open(log_file, 'r') as log:
            processed_files = log.read().splitlines()
        if file_name in processed_files:
            print(f"Skipping {file_name}, already processed.")
            return
    #print(file_name)
    Br_fits = fits.open(os.path.join(Br_PRED_DIR, file_name.replace("XX", "Br")))
    
    Br = Br_fits[1].data
    mask = get_data_from_fits(os.path.join(MASK_SAVE_DIR, file_name.replace("XX", "mask")))
    
    dataFix = None
    
    # If there are no holes, skip
    #print(np.sum(mask))
    if np.sum(mask) == 0:
        dataFix = Br
    else:
        # !!! dilate mask 
        mask = scipy.ndimage.binary_dilation(mask, np.ones((3, 3)))
        dataFix = quickInterpolate(Br, mask)
        
    if MODE == 'Overwrite':
        Br_fits[1].data = dataFix
        Br_fits.writeto(os.path.join(Br_PRED_DIR, file_name.replace("XX", "Br")), overwrite=True)
    else:
        pack_to_fits(SAVE_DIR, file_name_to_pack_fits(file_name), dataFix, Br_fits, 'spDisambig_Br', '', whether_flip = False)
    
    # Log the processed file
    with open(log_file, 'a') as log:
        log.write(f"{file_name}\n")    
    
    if True:
        if not os.path.exists(os.path.join(MASK_SAVE_DIR, 'hole_fix')):
            os.makedirs(os.path.join(MASK_SAVE_DIR, 'hole_fix'), exist_ok=True)
        #plt.imsave(os.path.join(MASK_SAVE_DIR, 'hole_fix', file_name.replace(".fits", "_dilated_mask.png")), mask)    
        #plt.imsave(os.path.join(MASK_SAVE_DIR, 'hole_fix', file_name.replace(".fits", "_data.png")), ssqrt(Br), vmin=-54, vmax=54, cmap='PuOr')
        #plt.imsave(os.path.join(MASK_SAVE_DIR, 'hole_fix', file_name.replace(".fits", "_dataFix.png")), ssqrt(dataFix), vmin=-54, vmax=54, cmap='PuOr')
    
    