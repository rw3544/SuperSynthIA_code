# Works for both pipeline and extra_layer



import numpy as np
import os
import sys
import numpy as np
import zarr
import torch
from model import *
import matplotlib.pyplot as plt
from dataset import *
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
from utils import *
from visualize_utils import get_hexbin_range

def evaluate_item_pipeline(
    Dataset_DIR:str, 
    count:int, 
    IMG_SAVE_PATH: str
    ):
    PRED_DIR = os.path.join(Dataset_DIR, f'{count}_predict.npy')
    TRUE_DIR = os.path.join(Dataset_DIR, f'{count}_true.npy')
    
    pred = np.load(PRED_DIR)
    true = np.load(TRUE_DIR)
    
def circular180Mean_vec(x):
    # x in (H,W,C)
    """Given data that's 180 ambiguous, average it without inducing a bias towards 90"""
    dataSin, dataCos = np.sin(np.deg2rad(x*2)), np.cos(np.deg2rad(x*2)) #lift to the unit circle
    avgSin, avgCos = np.mean(dataSin, axis=2), np.mean(dataCos, axis=2) # average vectors
    res = np.rad2deg(np.arctan2(avgSin, avgCos))/2 #splat back to the angle
    res[res<0] += 180
    return res #fix the domain






def circular180Mean(x):
    """Given data that's 180 ambiguous, average it without inducing a bias towards 90"""
    dataSin, dataCos = np.sin(np.deg2rad(x*2)), np.cos(np.deg2rad(x*2)) #lift to the unit circle
    avgSin, avgCos = np.mean(dataSin), np.mean(dataCos) # average vectors
    res = np.rad2deg(np.arctan2(avgSin, avgCos))/2 #splat back to the angle
    return res+180 if res < 0 else res #fix the domain

########
# R: pred_arr: DIR of predicted npy generated by predict_.py;
#    TRUE_arr: arr of GT label from zarr
# E: return (percentage, total) exclude the nans
########
def mae_item_percentage(pred_arr, TRUE_arr, t):
    
    assert(pred_arr.shape == TRUE_arr.shape)
    loss = np.abs(pred_arr-TRUE_arr)
    total = len(pred_arr)-np.sum(np.isnan(TRUE_arr))
    within = np.nansum(np.less(loss, t))

    return within/total, total

    
'''
########
# R: DATA_DIR: DIR of npys
#    img_idx_list: list of index of images to be sampled
#    GT: the zarr of GT 
#    t: the threshold
# E: array of shape [2, len(img_idx_list)]
#       arr[0] is the percentage
#       arr[1] is the num_pixel
########
def calculate_mae_percentage(
    DATA_DIR:str, 
    num_samples:int,
    GT_zarr_DIR:str,
    t):
    ########
    # Args:
    #   DATA_DIR: Dir where the npy are stored
    #   img_idx_list: list of image index (the index that corresponds to timestamp)
    #   t: threshold 
    # 
    #
    # Return:
    #   array of shape [2, len(img_idx_list)]
    #       arr[0] is the percentage
    #       arr[1] is the num_pixel
    #
    ########
    GT = zarr.open(GT_zarr_DIR, mode='r')
    timestamp_arr = np.load(os.path.join(DATA_DIR, 'TIMESTAMP.npy'))

    arr = np.zeros((2, num_samples))
    for i in range(num_samples):
        timestamp = timestamp_arr[i][0]
        PRED_DIR = os.path.join(DATA_DIR, f'{i}_predict_{timestamp}.npy')
        TRUE_arr = load_y(GT, i)
        tmp_percentage, num_pixel = mae_item_percentage(PRED_DIR, TRUE_arr, t)

        arr[0][i] = tmp_percentage
        arr[1][i] = num_pixel
    mae_PATH = os.path.join(DATA_DIR, 'mae_percent.npy')
    np.save(mae_PATH, arr)
    return arr

'''


# Takes in arr from calculate_mae_percentage and output the overall mae percentage
def calculate_combined_mae_percentage(arr):
    overall_sum = np.sum(arr[1])
    overall_percent = 0.
    for i in range(arr.shape[1]):
        overall_percent += arr[0][i]*arr[1][i]/overall_sum
    return overall_percent

def visualization_get_loss_list(
    arr, 
    img_idx_list:list):
    return arr[img_idx_list]



'''
def calculate_MSE_MAE_percentage_loss(
    DATA_DIR:str, 
    num_samples:int,
    GT_zarr_DIR:str,
    MASK_zarr_DIR:str,
    bin_class,
    t):
    ########
    # Args:
    #   DATA_DIR: Dir where the npy are stored
    #   img_idx_list: list of image index (the index that corresponds to timestamp)
    #   t: threshold 
    # 
    #
    # Return:
    #   arr of mse loss for each image
    #
    ########
    MSE_arr = []
    MAE_arr = []
    GT = zarr.open(GT_zarr_DIR, mode='r')
    timestamp_arr = np.load(os.path.join(DATA_DIR, 'TIMESTAMP.npy'))
    MASK_zarr = zarr.open(MASK_zarr_DIR, mode='r')
    
    percentage_arr = np.zeros((2, num_samples))
    
    
    
    for i in range(num_samples):
        timestamp = timestamp_arr[i][0]
        PRED_DIR = os.path.join(DATA_DIR, f'{i}_predict_{timestamp}.npy')
        TRUE_arr = load_y(GT, i)
        pred_arr = np.load(PRED_DIR)
        pred_arr = pred_arr.squeeze()
        dmask = load_eroded_mask(MASK_zarr, i)
        
        y_mask = ~np.isnan(TRUE_arr)
        dmask *= y_mask
        
        if bin_class != None:
            pred_arr, TRUE_arr, dmask = apply_downsample_bins(pred_arr, TRUE_arr, dmask, bin_class)
        
        TRUE_arr = TRUE_arr[dmask]
        pred_arr = pred_arr[dmask]
        #import matplotlib.pyplot as plt
        #plt.imsave("/nfs/turbo/fouheyUnrep/ruoyuw/comb_mask.png", dmask)
        
        
        
        total = len(pred_arr) - np.sum(np.isnan(TRUE_arr))
        mse = np.nansum((TRUE_arr-pred_arr)**2)/total
        mae = np.nansum(np.abs(TRUE_arr - pred_arr))/total
        MSE_arr.append(mse)
        MAE_arr.append(mae)
        
        tmp_percentage, num_pixel = mae_item_percentage(pred_arr, TRUE_arr, t)
        percentage_arr[0][i] = tmp_percentage
        percentage_arr[1][i] = num_pixel
        
    MSE_arr = np.array(MSE_arr)
    MAE_arr = np.array(MAE_arr)
    assert(len(MSE_arr) == num_samples)
    assert(len(MAE_arr) == num_samples)
    
    mse_PATH = os.path.join(DATA_DIR, 'MSE_LOSS.npy')
    mae_PATH = os.path.join(DATA_DIR, 'MAE_LOSS.npy')
    percentage_PATH = os.path.join(DATA_DIR, 'MAE_percentage.npy')
    
    if bin_class != None:
        mse_PATH = os.path.join(DATA_DIR, f'MSE_LOSS_{bin_class._get_description()}.npy')
        mae_PATH = os.path.join(DATA_DIR, f'MAE_LOSS_{bin_class._get_description()}.npy')
        percentage_PATH = os.path.join(DATA_DIR, f'MAE_percentage_{bin_class._get_description()}.npy')
    
    np.save(mse_PATH, MSE_arr)
    np.save(mae_PATH, MAE_arr)
    np.save(percentage_PATH, percentage_arr)
    
    return MSE_arr, MAE_arr, percentage_arr


'''

def per_img_MSE_MAE_percentage_loss(
    pred, 
    gt,
    dmask,
    bin_class,
    t):
    ########
    # Args:
    #   DATA_DIR: Dir where the npy are stored
    #   img_idx_list: list of image index (the index that corresponds to timestamp)
    #   t: threshold 
    # 
    #
    # Return:
    #   arr of mse loss for each image
    #
    ########
    if bin_class != None:
        pred, gt, dmask = apply_downsample_bins(pred, gt, dmask, bin_class)
        
    gt = gt[dmask]
    pred = pred[dmask]
        #import matplotlib.pyplot as plt
        #plt.imsave("/nfs/turbo/fouheyUnrep/ruoyuw/comb_mask.png", dmask)
    
    gt = gt.reshape(-1)
    pred = pred.reshape(-1)
    
    error = pred - gt
    MAE = np.mean(np.abs(error)) # average across the pixels of all the errors
    RMSE = np.sqrt(np.mean(error**2)) # root mean squared error
    PGP = np.mean(np.abs(error) < t)
    
    return MAE, RMSE, PGP



def complete_evaluation(DATA_DIR:str, 
    num_samples:int,
    GT_zarr_DIR:str,
    MASK_zarr_DIR:str,
    bin_class,
    t, 
    y,
    IMG_DIR,):
    
    description = 'standard'
    if bin_class != None:
        description = bin_class._get_description()
    
    flat_gt_DIR = os.path.join(DATA_DIR, f'flat_gt_{description}.npy')
    flat_pred_DIR = os.path.join(DATA_DIR, f'flat_pred_{description}.npy')
    
    flat_gt = []
    flat_pred = []
    if os.path.exists(flat_gt_DIR):
        flat_gt = np.load(flat_gt_DIR)
        flat_pred = np.load(flat_pred_DIR)
    else:
        # create flatten prediction and ground truth
    
        GT = zarr.open(GT_zarr_DIR, mode='r')
        timestamp_arr = np.load(os.path.join(DATA_DIR, 'TIMESTAMP.npy'))
        MASK_zarr = zarr.open(MASK_zarr_DIR, mode='r')
    
    
    
        for i in range(num_samples):
            timestamp = timestamp_arr[i][0]
            PRED_DIR = os.path.join(DATA_DIR, f'{i}_predict_{timestamp}.npy')
            TRUE_arr = load_y(GT, i)
            pred_arr = np.load(PRED_DIR)
            pred_arr = pred_arr.squeeze()
            dmask = load_eroded_mask(MASK_zarr, i)
            
            y_mask = ~np.isnan(TRUE_arr)
            dmask *= y_mask
            pred_mask = ~np.isnan(pred_arr)
            dmask *= pred_mask
            
            
            if bin_class != None:
                if 'Azimuth' in GT_zarr_DIR:
                    bin_class._set_Azimuth()
                pred_arr, TRUE_arr, dmask = apply_downsample_bins(pred_arr, TRUE_arr, dmask, bin_class)
            
            TRUE_arr = TRUE_arr[dmask]
            pred_arr = pred_arr[dmask]
            #import matplotlib.pyplot as plt
            #plt.imsave("/nfs/turbo/fouheyUnrep/ruoyuw/comb_mask.png", dmask)
            
            TRUE_arr = TRUE_arr.reshape(-1)
            pred_arr = pred_arr.reshape(-1)
            
            flat_gt.append(TRUE_arr)
            flat_pred.append(pred_arr)
        
        flat_gt = np.hstack(flat_gt)
        flat_pred = np.hstack(flat_pred)
        
        np.save(flat_gt_DIR, flat_gt)
        np.save(flat_pred_DIR, flat_pred)
        
    error = flat_pred - flat_gt
    
    
    MAE = np.mean(np.abs(error)) # average across the pixels of all the errors
    RMSE = np.sqrt(np.mean(error**2)) # root mean squared error
    PGP = np.mean(np.abs(error) < t)

    # Create hexbin plot

    cmap_val, vmin_val, vmax_val = get_hexbin_range(y)
    plt.figure(dpi=300)
    plt.hexbin(flat_gt, flat_pred, gridsize=200, bins='log', extent=[vmin_val,vmax_val,vmin_val,vmax_val])
    plt.axis('square')
    plt.axline([0,0], slope=1, color='black')
    plt.title(f"Histogram of gt versus prediction ({description})")
    plt.colorbar()
    plt.savefig(os.path.join(IMG_DIR, f"hexbin_{description}.png"))
    plt.close()

    return MAE, RMSE, PGP


# Downsample the image for evaluation
class downsample_img():
    def __init__(self, k:int):
        super().__init__()
        self.k = k
        self.description = f'downsample_{k}'
        self.Azimuth = False 
    
    def _set_Azimuth(self):
        print('Azimuth Circular binning')
        self.Azimuth = True 
    
    def _get_binned_img(self, img, mask:bool):
        if mask == True or self.Azimuth == False:
            H,W = img.shape
            new_H = H//self.k
            new_W = W//self.k
            binned = np.zeros((new_H, new_W))
            for i in range(self.k):
                for j in range(self.k):
                    fraction = img[i:new_H*self.k:self.k, j:new_W*self.k:self.k]
                    binned += fraction/float(self.k*self.k)
            return binned
        elif self.Azimuth == True:
            H,W = img.shape
            new_H = H//self.k
            new_W = W//self.k
            binned = np.zeros((new_H, new_W, self.k**2))
            for i in range(self.k):
                for j in range(self.k):
                    fraction = img[i:new_H*self.k:self.k, j:new_W*self.k:self.k]
                    binned[:,:,i*self.k+j] = fraction
            return circular180Mean_vec(binned)
    
    def _get_description(self):
        return self.description



def uniqueLatLonBins(lat, lon, latBin, lonBin, dmask):
    #figure out what's invalid
    invalidMask = np.isnan(lat) | np.isnan(lon) | np.isinf(lat) | np.isinf(lon) | np.logical_not(dmask)

    #change invalid to the median value to avoid issues with unique and nans
    lat[invalidMask] = np.nanmedian(lat)
    lon[invalidMask] = np.nanmedian(lon)


    #print(np.nanmedian(lat))
    #print(np.nanmedian(lon))
    
    #round to latBin
    latRound = np.floor(lat/latBin).astype(int)
    lonRound = np.floor(lon/lonBin).astype(int)
    
    #compute the unique values and then an index into them
    latVals, latValIndex = np.unique(latRound, return_inverse=True)
    lonVals, lonValIndex = np.unique(lonRound, return_inverse=True)

    #reshape, then form the lat/lon bins, then find the linear index for the lat+lon bin
    latValIndex = latValIndex.reshape(latRound.shape)
    lonValIndex = lonValIndex.reshape(latRound.shape)
    latLonValIndex = latValIndex*(np.max(lonValIndex)+1)+lonValIndex
    # If negative exists, max is not enough, 2*max should be ok

    
    #make the counts; we can't use np.unique() since the invalid ones are set to the 
    #median, screwing up the counts.
    latValCount = np.bincount(latValIndex[~invalidMask])
    lonValCount = np.bincount(lonValIndex[~invalidMask])
    latLonValCount = np.bincount(latLonValIndex[~invalidMask])
    
    

    #make a Nx2 latLon look up table for lon/lat, and then also N for lon and N for lat
    latLonN = latLonValCount.size
    latLonInds = np.arange(latLonN).reshape(latLonN,1)

    #undo the binning
    latVals = latVals*latBin
    lonVals = lonVals*lonBin
    latLonVals = np.hstack([ latVals[latLonInds//(np.max(lonValIndex)+1)], lonVals[latLonInds%(np.max(lonValIndex)+1)] ])

    #mark invalid lat/lon pixels as -1
    lonValIndex[invalidMask] = -1
    latValIndex[invalidMask] = -1
    latLonValIndex[invalidMask] = -1

    info = {
        'latValIndex': latValIndex, 'lonValIndex': lonValIndex, 'latLonValIndex': latLonValIndex, #HxW indices for binned data
        'latN': latVals.size, 'lonN': lonVals.size, 'latLonN': latLonN, #number of bin indices (i.e., latValIndex.max()
        'latVals': latVals, 'lonVals': lonVals, 'latLonVals': latLonVals, #the values latVals[latValIndex] should be close to lat
        'latValCount': latValCount, 'lonValCount': lonValCount, 'latLonValCount': latLonValCount #the pixel counts for each bin
    }
    return info

class longitude_latitude_bin():
    def __init__(self, latBin, lonBin):
        super().__init__()
        self.latBin = latBin
        self.lonBin = lonBin
        self.description = f'latBin_{latBin}_lonBin_{lonBin}'
        self.Azimuth = False  
    
    def _set_Azimuth(self):
        print('Azimuth Circular binning')
        self.Azimuth = True 
    
    def _get_binned_img(self, pred, gt, dmask, DIR):
        data = np.load(DIR)
            
        helioCoord = data['hmiMetaHeliographic']
        lon, lat = helioCoord[:,:,0], helioCoord[:,:,1]
        info = uniqueLatLonBins(lat, lon, self.latBin, self.lonBin, dmask)
        
        suppress = np.where(info['latLonValCount'] < 50)
        onlyBigBins_latLonValIndex = info['latLonValIndex'].copy()
        onlyBigBins_latLonValIndex[np.isin(onlyBigBins_latLonValIndex,suppress)] = -1
        
        
        binned_pred = []
        binned_gt = []
        
        uniq_idx_remain = np.unique(onlyBigBins_latLonValIndex)
        for uniq_val in uniq_idx_remain:
            if uniq_val != -1:
                temp_mask = (onlyBigBins_latLonValIndex == uniq_val)

                
                if self.Azimuth == False:
                    binned_pred.append(np.mean(pred[temp_mask]))
                    binned_gt.append(np.mean(gt[temp_mask]))
                else:
                    binned_pred.append(circular180Mean(pred[temp_mask].reshape(-1)))
                    binned_gt.append(circular180Mean(gt[temp_mask].reshape(-1)))
       

        binned_pred = np.hstack(binned_pred)
        binned_gt = np.hstack(binned_gt)
        
        
        return binned_pred, binned_gt
        
    def _get_description(self):
        return self.description

