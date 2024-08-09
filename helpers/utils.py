import os, re
import numpy as np
import zarr
import torch
import scipy.ndimage

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




def triangular_distribution(arr_shape, device, seed = None):
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            
    uni = torch.rand(arr_shape, device=device)
    X = torch.zeros_like(uni, device=device)
    gmask = (uni >= 0.5)
    lmask = torch.logical_not(gmask)
    
    X[lmask] = np.sqrt(6) * (-1 + torch.sqrt(2*uni[lmask]))
    X[gmask] = -(np.sqrt(6)*(-12 + torch.sqrt(-288*uni[gmask] + 288))) / 12 
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