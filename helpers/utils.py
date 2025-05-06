import os, re, threading, shutil
import numpy as np
import zarr
import torch
import scipy.ndimage

import astropy.io.fits as fits 

def isLocked(p):
    if os.path.exists(p):
        return True
    try:
        os.mkdir(p)
        return False
    except:
        return True

# Verify if the saved zarr is the same as the original data
def verify_zarr(
    original,
    file_dir: str,
    ):
    ########
    # Important Args:
    # 
    # original: original data, usually numpy array
    # file_dir: directory for saved data
    ########
    loaded = zarr.load(file_dir)
    ret = np.array_equal(original, loaded, equal_nan=True)
    print('Zarr Verified')
    assert(ret)
    return ret
    

# Retrieve original array from padded ones 
def de_pad(
    padded,
    ):
    # H,W,C
    x_h = None
    x_w = None
    if len(padded.shape) == 2:
        x_h = int(padded[-1,0])
        x_w = int(padded[-1,1])    
    elif len(padded.shape) == 3:
        x_h = int(padded[-1,0,0])
        x_w = int(padded[-1,1,0])  
    
    recovered = padded[:x_h, :x_w]
    return recovered


def get_path(
    base:str,
    type:str,
    ):
    return os.path.join(base, f'{type}.zarr')


def save_checkpoint(model, epoch_num, CHECKPOINT_DIR):
    state = {
        "epoch":epoch_num,
        "state_dict":model.state_dict(),
    }
    torch.save(state, os.path.join(CHECKPOINT_DIR, f'epoch={epoch_num}.checkpoint.pth.tar'))
    print(f'Epoch = {epoch_num}, model saved')
    
def bins_to_output(pred, max_divisor):
    pred = torch.nn.functional.softmax(pred.squeeze(), dim=0)

    # find the max probability bin
    _, max_indices = torch.max(pred, 0)
    max_indices = max_indices.unsqueeze(0)

    # make an ordinal scatter against the one hot args.bins
    max_mask = torch.zeros((80, pred.shape[1], pred.shape[2])).to(0)
    scatter_ones = torch.ones(max_indices.shape).to(0)
    scatter_range = torch.arange(80).unsqueeze(1).unsqueeze(1).float().to(0)

    up_max_indices = (max_indices+1).clamp(0, 80-1)
    down_max_indices = (max_indices-1).clamp(0, 80-1)

    mod_max_indices = max_mask.scatter_(0, max_indices, scatter_ones)
    mod_max_indices = mod_max_indices.scatter_(0, up_max_indices, scatter_ones)
    mod_max_indices = mod_max_indices.scatter_(0, down_max_indices, scatter_ones)

    masked_probabilities = (mod_max_indices * pred)
    normed_probabilities = masked_probabilities / masked_probabilities.sum(dim=0) 
    indices = (normed_probabilities * scatter_range).sum(dim=0)
    pred_im = (indices.float() / ((80-1) / max_divisor)).cpu()
    return pred_im.cpu()


# Takes in ZARR of Y, index and return the correct y
# Return: y (numpy arr) (H,W)
def load_y(GT, i):
    true = GT[i]
    y = (torch.from_numpy(true).float())
    y = y.permute(1,2,0)
    y = de_pad(y)
    y = y.squeeze()
    return y.numpy()



# Return 
#   dmask: (H,W) eroded disk mask
def load_eroded_mask(mask_zarr, i:int):
    dmask_item = mask_zarr[i]
    dmask = (torch.from_numpy(dmask_item).float())
    dmask = dmask.permute(1,2,0)
    dmask = de_pad(dmask)
    dmask = dmask.squeeze()
    dmask = dmask.bool()
    
    dmask = scipy.ndimage.binary_erosion(dmask, structure=np.ones((3,3)), iterations=2)
    
    return dmask 


def apply_downsample_bins(pred, true, dmask, bin_class):
    pred = bin_class._get_binned_img(pred, mask=False)
    true = bin_class._get_binned_img(true, mask=False)
    dmask = dmask.astype(int)
    dmask = bin_class._get_binned_img(dmask, mask = True)
    dmask[dmask < 1] = 0
    dmask = dmask.astype(bool)
    return pred, true, dmask


def apply_lat_lon_bins(DIR, pred, gt, dmask, bin_class):
    binned_pred, binned_gt = bin_class._get_binned_img(pred, gt, dmask, DIR) 
    return binned_pred, binned_gt

def utils_make_dir(DIR:str):
    if not os.path.exists(DIR):
        os.makedirs(DIR)

def confidence_Interval_simple(bins, prob, percent):
    # percent in (0, 1) 
    assert(percent > 0 and percent < 1)
    assert(len(bins) == len(prob))
    
    
    percent_each_side = (1 - percent) / 2
    left_total_prob = 0
    
    left_val = 0
    for i in range(len(bins)):
        b = bins[i]
        left_total_prob += prob[i]
        if left_total_prob == percent_each_side:
            left_val == b
            break
        elif left_total_prob > percent_each_side:
            if i == 0:
                left_val = 0
                break
            else:
                lower_i = i - 1
                upper_prob = prob[i]
                l_upper_total_prob = left_total_prob
                l_lower_total_prob = left_total_prob - upper_prob
                
                # lower * (1-x) + upper *x = percent_each_side
                # -lower * x + lower + upper * x = percent_each_side
                # (upper - lower)x = percent_each_side - lower
                left_dist = (percent_each_side - l_lower_total_prob)/(l_upper_total_prob - l_lower_total_prob)
                left_dist *= (bins[i] - bins[lower_i])
                left_val = bins[lower_i] + left_dist
                break
    
    right_val = 0 
    right_total_prob = 0
    for i in np.arange(len(bins)-1, -1, -1):
        b = bins[i]
        right_total_prob += prob[i]
        if right_total_prob == percent_each_side:
            right_val = b 
            break
        elif right_total_prob > percent_each_side:
            if i == len(bins) - 1:
                right_val = bins[i]
                break
            else:
                upper_i = i + 1
                lower_prob = prob[i]
                r_lower_total_prob = right_total_prob
                r_upper_total_prob = right_total_prob - lower_prob
                right_dist = 1 - (percent_each_side - r_upper_total_prob)/(r_lower_total_prob - r_upper_total_prob)
                right_dist *= (bins[upper_i] - bins[i])
                right_val = bins[i] + right_dist
                break
    return left_val, right_val

# Create a global lock
seed_lock = threading.Lock()


def triangular_distribution(arr_shape, device, seed=None):
    with seed_lock:
        with torch.random.fork_rng():
            if seed is not None:
                # Set the new seed inside the context
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                    
            # Generate random numbers
            uni = torch.rand(arr_shape, device=device)
            X = torch.zeros_like(uni, device=device)
            gmask = (uni >= 0.5)
            lmask = torch.logical_not(gmask)

            X[lmask] = np.sqrt(6) * (-1 + torch.sqrt(2*uni[lmask]))
            X[gmask] = -(np.sqrt(6)*(-12 + torch.sqrt(-288*uni[gmask] + 288))) / 12 

        # RNG state is automatically restored after the with block
    return X



def extract_timestamp_from_fits(filename):
    # Define the regex pattern to match the date and time in the filename
    pattern = r'\d{8}_\d{6}'
    
    # Search for the pattern in the filename
    match = re.search(pattern, filename)
    
    if match:
        # Extract the matched string
        timestamp = match.group(0)
        return timestamp
    else:
        raise ValueError("Timestamp not found in the filename")



#region -------------------------------- Fits File Handling --------------------------------
def get_data_from_fits(fits_path):
    # !!! Loads predictions saved as fits, not for IQUV
    # Should output float value
    
    # 1) Open FITS and grab extension 1
    hdul = fits.open(fits_path)           # returns an HDUList               
    hdu  = hdul[1]                         # Primary=0, first ImageHDU=1    

    # 2) Extract raw data and header
    raw_data = hdu.data                    # NumPy int32 array             
    hdr      = hdu.header                  # FITS header                    

    # 3) Read scaling keywords (with defaults)
    bscale = hdr.get('BSCALE', 1.0)        # scale factor                    
    bzero  = hdr.get('BZERO', 0.0)         # offset                           
    blank  = hdr.get('BLANK', None)        # blank pixel indicator      
    breakpoint()
    
    # 4) Convert blanked pixels to NaN
    if blank is not None:
        raw_data[raw_data == blank] = np.nan  # mark undefined pixels as NaN    
    
    # 5) Apply linear transform: physical = raw * BSCALE + BZERO
    data = raw_data.astype(np.float32) * bscale + bzero

    # 6) Cleanup and return
    hdul.close()                           # free file resources           
    return data


def get_IQUV_from_fits(fits_dir):
    # !!! Loads IQUV & mask & uncertainty from fits, not for predictions like Br, Bp, ...
    # Should output float value
    arr = fits.open(fits_dir)[1].data
    return arr





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
        
    # Scale data according to JSOC standard
    # 1) Determine scaling parameters
    if y_name == 'spInv_Stray_Light_Fill_Factor':
        bscale = 0.0001
    else:
        bscale = 0.01
    bzero = 0.0
    blank = np.iinfo(np.int32).min  # −2147483648
    
    # 2) Scale and convert to int32, mapping NaNs to BLANK
    scaled = np.round((imageData - bzero) / bscale).astype(np.int32)
    scaled[np.isnan(imageData)] = blank
    imageData_int = scaled

    # 3) Prepare header with scaling keywords
    header0 = (headerSrc[0].header).copy()
    for key in ('BZERO','BSCALE','BLANK'):
        header0.pop(key, None)
    
    
    primary = fits.PrimaryHDU(data=None, header=None)
    image_hdu = fits.ImageHDU(data=imageData_int, header=header0)
    hdul = fits.HDUList([primary, image_hdu])
    
    hdr = hdul[1].header
    hdr['BZERO']  = (bzero, 'Real data offset')              
    hdr['BSCALE'] = (bscale, 'Real-to-int scale factor')
    hdr['BLANK']  = (int(blank), 'Undefined pixel value')

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




#endregion ---------------------------------------------------------------------------------