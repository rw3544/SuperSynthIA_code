import torch, hashlib
from helpers.utils import *




# Get the std of prediction
def decoder_std(n_prob:torch.tensor, decoded_val:torch.tensor, bins:torch.tensor):
    # n_prob: K x H x W
    # decoded_val: H, W
    # bins: K
    K,H,W = n_prob.shape
    assert(decoded_val.shape == (H,W))
    assert(bins.shape == (K,))
    bins = bins.unsqueeze(1).unsqueeze(2)
    pred_std = torch.sqrt(torch.sum(n_prob*(bins - decoded_val.unsqueeze(0))**2, dim=0))
    return pred_std
    

def encoder(value:torch.tensor, bins:torch.tensor):
    ########
    # Args:
    #
    #   value: Tensor of values (N,C,H,W) N=C=1
    #   bins: Tensor of bins (Kx1)
    #   
    #
    # Return:
    #   prob: Tensor of probability (N,K,H,W)
    ########
    assert(torch.is_tensor(value) and torch.is_tensor(bins))
    GLOBAL_DEVICE = value.device
    bins = bins.to(GLOBAL_DEVICE)
    
    # First convert value to HxW
    value = torch.squeeze(value).float()
    assert(len(value.size()) == 2)
    assert(len(bins.size()) == 1)
    K, = bins.size()
    H,W = value.size()
    
    
    MAX_BIN_VAL = torch.max(bins) 
    value = torch.clamp_max(value, MAX_BIN_VAL)
    
    MIN_BIN_VAL = torch.min(bins)
    value = torch.clamp_min(value, MIN_BIN_VAL)
    
    
    to_minus = torch.reshape(bins.clone(), (K,1,1)).float()
    to_minus = torch.broadcast_to(to_minus, (K, H, W))
    distance = value - to_minus
    
    
    # Set <0 value to maximum value for the left boundary
    distance_no_negative = distance.clone()
    distance_no_negative[distance_no_negative < 0] = 99999999999
    # Find minimum positive value
    left_val, left_idx = torch.min(distance_no_negative, dim=0)
    
    
    # Set >0 value to minimum value for the right boundary
    distance_no_positive = distance.clone()
    distance_no_positive[distance_no_positive > 0] = -99999999999
    # Find maximum negative value
    right_val, right_idx = torch.max(distance_no_positive, dim=0)
    
    
    on_bin_mask = (left_idx == right_idx)
    
    
    # prob: KxHxW
    # The probability matrix that will be returned
    prob = torch.zeros_like(distance)
    
    
    # bin_distance: the distance between bins
    # for on_bin
    bin_distance = torch.abs(left_val) + torch.abs(right_val)
    bin_distance[on_bin_mask] = -1
    
    left_prob = torch.abs(right_val)/bin_distance
    right_prob = torch.abs(left_val)/bin_distance
    
    
    # Set prob of on-bin value to 1
    left_prob[on_bin_mask] = 1.
    right_prob[on_bin_mask] = 1.
    
    
    prob[left_idx, torch.arange(H)[:,None], torch.arange(W)] = left_prob
    prob[right_idx, torch.arange(H)[:,None], torch.arange(W)] = right_prob
    prob = torch.unsqueeze(prob, dim=0)
    prob += 1e-8
    
    return prob


# No softmax, just normalize into probability and calculate expected value
def decoder_basic(prob:torch.tensor, bins:torch.tensor):
    ########
    # Args:
    #
    #   prob: Tensor of probability (N,K,H,W)
    #   bins: Tensor of bins (Kx1)
    #   
    #
    # Return:
    #   value: Tensor of values (N,C,H,W) N=C=1
    ########
    assert(torch.is_tensor(prob) and torch.is_tensor(bins))
    GLOBAL_DEVICE = prob.device
    bins = bins.to(GLOBAL_DEVICE)
    
    
    # First convert value to HxW
    prob = torch.squeeze(prob).float()
    assert(len(prob.size()) == 3)
    assert(len(bins.size()) == 1)
    K, = bins.size()
    C,H,W = prob.size()
    assert(C == K)
    
    # Normalize Probability
    prob_sum = torch.sum(prob, dim=0, keepdim=True)
    norm_prob = prob/prob_sum
    bin_matrix = torch.ones_like(prob)*(bins.reshape(K,1,1))
    
    
    value = torch.sum(norm_prob*bin_matrix, dim=0)
    value = torch.unsqueeze(value, dim=0)
    value = torch.unsqueeze(value, dim=0)
    return value


# Only look at the most likely bin
def decoder_most_likely(prob:torch.tensor, bins:torch.tensor):
    ########
    # Args:
    #
    #   prob: Tensor of probability (N,K,H,W)
    #   bins: Tensor of bins (Kx1)
    #   
    #
    # Return:
    #   value: Tensor of values (N,C,H,W) N=C=1
    ########
    assert(torch.is_tensor(prob) and torch.is_tensor(bins))
    GLOBAL_DEVICE = prob.device
    bins = bins.to(GLOBAL_DEVICE)
    
    # First convert value to #bins x H x W
    prob = torch.squeeze(prob).float()
    prob = torch.softmax(prob, dim=0)
    
    assert(len(prob.size()) == 3)
    assert(len(bins.size()) == 1)
    K, = bins.size()
    C,H,W = prob.size()
    assert(C == K)
    
    # max_val, max_idx is in (H,W)
    max_val, max_idx = torch.max(prob, dim=0)

    max_up_idx = (max_idx + 1).clamp(0, K-1)
    max_up_val = prob[max_up_idx, torch.arange(H)[:, None], torch.arange(W)]
    
    max_down_idx = (max_idx - 1).clamp(0, K-1)
    max_down_val = prob[max_down_idx, torch.arange(H)[:, None], torch.arange(W)]
    
    # deal with edge bin
    up_edge_mask = (max_up_idx == max_idx)
    down_edge_mask = (max_down_idx == max_idx)
    
    
    max_up_val[up_edge_mask] = -1
    max_down_val[down_edge_mask] = -1
    
    second_max_val = torch.maximum(max_up_val, max_down_val)
    second_max_idx = max_up_idx.clone()
    second_max_idx[max_up_val < max_down_val] = max_down_idx[max_up_val < max_down_val]
    
    # Normalize the probability again
    val_sum = max_val + second_max_val
    max_val = max_val/val_sum
    second_max_val = second_max_val/val_sum
    
    decoded_val = max_val*bins[max_idx] + second_max_val*bins[second_max_idx]
    return decoded_val
    

# Take the median bin
def decoder_median(prob:torch.tensor, bins:torch.tensor):
    ########
    # Args:
    #
    #   prob: Tensor of probability (N,K,H,W)
    #   bins: Tensor of bins (Kx1)
    #   
    #
    # Return:
    #   value: Tensor of values (N,C,H,W) N=C=1
    ########
    assert(torch.is_tensor(prob) and torch.is_tensor(bins))
    GLOBAL_DEVICE = prob.device
    bins = bins.to(GLOBAL_DEVICE)
    
    # First convert value to #bins x H x W
    prob = torch.squeeze(prob).float()
    prob = torch.softmax(prob, dim=0)
    
    assert(len(prob.size()) == 3)
    assert(len(bins.size()) == 1)
    K, = bins.size()
    C,H,W = prob.size()
    assert(C == K)
    
    # Find the mean bin, first let all <0.5 prob to be very large and find the min
    cumu_prob = torch.cumsum(prob, dim=0)
    
    
    cumu_prob[cumu_prob < 0.5] = 999 
    over_50_val, over_50_idx = torch.min(cumu_prob, dim=0)
    down_50_idx = over_50_idx.clone()
    
    # down_50_idx is just over_50_idx - 1
    # Mind -1, take -1 to 1
    down_50_idx -= 1 
    down_50_idx[down_50_idx == -1] = 1 
    down_50_val = prob[down_50_idx, torch.arange(H)[:, None], torch.arange(W)]

    
    # Normalize the probability again
    val_sum = over_50_val + down_50_val
    over_50_val = over_50_val/val_sum
    down_50_val = down_50_val/val_sum
    
    decoded_val = over_50_val*bins[over_50_idx] + down_50_val*bins[down_50_idx]
    return decoded_val
    

# Given a key, hash the fixed noise determistically (Fisher-Yates)
# Very slow (~15 minutes per 4096x4096)
def permute_tensor_deterministic_slow(combined_string, fixed_noise):
    C, H, W = fixed_noise.shape
    flat_noise = fixed_noise.reshape(-1)
    num_elements = flat_noise.size(0)
    
    # Step 2: Generate multiple hash values to create a larger pool of random values
    hash_values = []
    num_hashes = (num_elements // 32) + 1  # Ensure enough hash values
    for i in range(num_hashes):
        hash_object = hashlib.sha256((combined_string + str(i)).encode())
        hash_hex = hash_object.hexdigest()
        # Map hash values to the range [0, len(flat_noise))
        for j in range(0, len(hash_hex), 8):
            hash_value = int(hash_hex[j:j+8], 16) % num_elements
            hash_values.append(hash_value)
    
    # Step 3: Generate a permutation of the tensor indices
    indices = list(range(flat_noise.size(0)))
    
    # Use the hash values to shuffle the indices using Fisher-Yates algorithm
    for i in range(len(indices) - 1, 0, -1):
        swap_idx = hash_values[i % len(hash_values)] % (i + 1)
        indices[i], indices[swap_idx] = indices[swap_idx], indices[i]
    flat_permuted_tensor = flat_noise[indices]
    permuted_tensor = flat_permuted_tensor.reshape(C, H, W)
    
    return permuted_tensor


def permute_tensor_deterministic_numpy(combined_string, fixed_noise):
    C, H, W = fixed_noise.shape
    flat_noise = fixed_noise.reshape(-1)
    num_elements = flat_noise.size(0)

    # Generate a single hash value
    hash_object = hashlib.sha256(combined_string.encode())
    seed = int(hash_object.hexdigest(), 16) % (2**32)  # Use 32-bit seed for the RNG

    # Use the hash value as the seed for the RNG
    rng = np.random.default_rng(seed)

    # Generate a permutation of the tensor indices
    indices = np.arange(num_elements)
    rng.shuffle(indices)

    # Apply the permutation
    flat_permuted_tensor = flat_noise[indices]
    permuted_tensor = flat_permuted_tensor.reshape(C, H, W)

    return permuted_tensor

def permute_tensor_deterministic_fast(combined_string, fixed_noise):
    C, H, W = fixed_noise.shape
    flat_noise = fixed_noise.reshape(-1)
    num_elements = flat_noise.size(0)

    # Generate a single hash value
    hash_object = hashlib.sha256(combined_string.encode())
    seed = int(hash_object.hexdigest(), 16) % (2**32)  # Use 32-bit seed for the RNG

    # Use the hash value as the seed for the RNG
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Generate a permutation of the tensor indices
    indices = torch.randperm(num_elements, device='cuda')
    
    # Apply the permutation
    flat_permuted_tensor = flat_noise[indices]
    permuted_tensor = flat_permuted_tensor.reshape(C, H, W)
    return permuted_tensor


#   fixed_noise is already on GLOBAL_DEVICE
def triangular_noise_generation_packer_through_fixed_noise(orig_prob_shape, reproducible, fixed_noise, hashing_key, device):
    C,H,W = orig_prob_shape
    if reproducible == False:
        noise = triangular_distribution(orig_prob_shape, device=device)
        return noise
    else:
        '''
        # For filling in 64x64 by 64x64
        # Use fixed_noise(81, 64, 64) to cover the whole image
        fixed_noise = fixed_noise[:C, :, :]
        TIMESTAMP, UP_LEFT_H, UP_LEFT_W = hashing_key
        
        # Calculate the number of repeats needed
        repeat_h = H // 64
        repeat_w = W // 64
        
        # Initialize an empty tensor with the desired shape
        enlarged_noise = torch.empty(C, H, W, device=device)

        # Double for loop to fill each portion individually
        for i in range(repeat_h):
            for j in range(repeat_w):
                # Calculate the coordinates of the upper-left corner
                c_up_left_h = i * 64
                c_up_left_w = j * 64
                
                comb_up_left_h = UP_LEFT_H + c_up_left_h
                comb_up_left_w = UP_LEFT_W + c_up_left_w
                combined_string = f"{TIMESTAMP}_{comb_up_left_h}_{comb_up_left_w}"
                
                permuted_noise = permute_tensor_deterministic_fast(combined_string, fixed_noise.clone())
                
                # Fill the portion of the tensor
                enlarged_noise[:, c_up_left_h:c_up_left_h + 64, c_up_left_w:c_up_left_w + 64] = permuted_noise

        return enlarged_noise
        '''
        # Use fixed_noise(81, 64, 64) to cover the whole image
        fixed_noise = fixed_noise[:C, :, :]
        TIMESTAMP, UP_LEFT_H, UP_LEFT_W = hashing_key
        
        # Calculate the number of repeats needed
        repeat_h = H // 64
        repeat_w = W // 64
        
        # Initialize an empty tensor with the desired shape
        enlarged_noise = fixed_noise.repeat(1, repeat_h, repeat_w)
        
        combined_string = f"{TIMESTAMP}_{UP_LEFT_H}_{UP_LEFT_W}"
        permuted_noise = permute_tensor_deterministic_fast(combined_string, enlarged_noise)
        return permuted_noise


#   fixed_noise is already on GLOBAL_DEVICE
def triangular_noise_generation_packer_through_torch_seed(orig_prob_shape, reproducible, hashing_key, device):
    C,H,W = orig_prob_shape
    if reproducible == False:
        noise = triangular_distribution(orig_prob_shape, device=device)
        return noise
    else:
        TIMESTAMP, UP_LEFT_H, UP_LEFT_W = hashing_key
        combined_string = f"{TIMESTAMP}_{UP_LEFT_H}_{UP_LEFT_W}"
        hash_object = hashlib.sha256(combined_string.encode())
        seed = int(hash_object.hexdigest(), 16) % (2**32)
        
        noise = triangular_distribution(orig_prob_shape, device=device, seed=seed)
        return noise
    



def decoder_median_with_noise_base(prob:torch.tensor, bins:torch.tensor, whether_CI = False, 
                                   whether_std = False, reproducible = False, fixed_noise = None,
                                   hashing_key = None):
    ########
    # Args:
    #
    #   prob: Tensor of probability (N,K,H,W)
    #   bins: Tensor of bins (Kx1)
    #   reproducible: if False, use random triangular noise in the shape of orig_prob
    #                 if True, use 81x64x64 saved fix noise & slide across (!!! size divible by 64)
    #   fixed_noise: use only if reproducible = True, the pre-generated noise to ensure reproducibility
    #   hashing_key: use only if reproducible = True, the key used to permute the fixed_noise
    #                expect to be (TIMESTAMP, UP_LEFT_H, UP_LEFT_W)
    #   
    #
    # Return:
    #   value: Tensor of values (N,C,H,W) N=C=1
    ########
    assert(torch.is_tensor(prob) and torch.is_tensor(bins))
    GLOBAL_DEVICE = prob.device
    bins = bins.to(GLOBAL_DEVICE)
    
    # First convert value to #bins x H x W
    orig_prob = torch.squeeze(prob).float()
    triangle_noise = triangular_noise_generation_packer_through_torch_seed(orig_prob.shape, reproducible, hashing_key, device=GLOBAL_DEVICE)
    n_prob = orig_prob.clone() + triangle_noise
    n_prob = torch.softmax(n_prob.clone(), dim=0)
    c_prob = torch.softmax(orig_prob.clone(), dim=0)
    
    assert(len(c_prob.size()) == 3)
    assert(len(bins.size()) == 1)
    K, = bins.size()
    C,H,W = c_prob.size()
    assert(C == K)
    
    # Find the mean bin, first let all <0.5 prob to be very large and find the min
    cumu_prob = torch.cumsum(c_prob, dim=0)
    
    
    cumu_prob[cumu_prob < 0.5] = 999 
    trash, on_50_idx = torch.min(cumu_prob, dim=0)
    down_50_idx = on_50_idx.clone()
    up_50_idx = on_50_idx.clone()
    
    # down_50_idx is just over_50_idx - 1
    # Mind -1, take -1 to 1
    down_50_idx -= 1 
    down_edge_mask = (down_50_idx == -1) 
    down_50_idx[down_edge_mask] = 1

    up_50_idx += 1
    up_edge_mask = (up_50_idx >= K)
    up_50_idx[up_edge_mask] = 1 

 
    u_on_50_val = n_prob[on_50_idx, torch.arange(H)[:, None], torch.arange(W)]
    u_down_50_val = n_prob[down_50_idx, torch.arange(H)[:, None], torch.arange(W)]
    

    d_on_50_val = n_prob[on_50_idx, torch.arange(H)[:, None], torch.arange(W)]
    d_up_50_val = n_prob[up_50_idx, torch.arange(H)[:, None], torch.arange(W)]
 
   
    o_on_50_val = n_prob[on_50_idx, torch.arange(H)[:, None], torch.arange(W)]
    o_down_50_val = n_prob[down_50_idx, torch.arange(H)[:, None], torch.arange(W)]
    o_up_50_val = n_prob[up_50_idx, torch.arange(H)[:, None], torch.arange(W)]

    # Calculate total sum
    val_sum = o_on_50_val + o_down_50_val + o_up_50_val


    o_down_50_val = o_down_50_val/val_sum
    o_on_50_val = o_on_50_val/val_sum
    o_up_50_val = o_up_50_val/val_sum 
    

    decoded_val = o_down_50_val* bins[down_50_idx] + o_on_50_val* bins[on_50_idx] + o_up_50_val* bins[up_50_idx]
    
    
    # Deal with edge cases by overwrite according to masks

    # Reach upper limit
    u_edge_sum = u_on_50_val + u_down_50_val
    u_on_50_val = u_on_50_val/u_edge_sum
    u_down_50_val = u_down_50_val/ u_edge_sum

    u_decode_val = u_down_50_val*bins[down_50_idx] + u_on_50_val* bins[on_50_idx]
    
    decoded_val[up_edge_mask] = u_decode_val[up_edge_mask]


    # Reach lower limit
    d_edge_sum = d_on_50_val + d_up_50_val
    d_on_50_val = d_on_50_val/d_edge_sum
    d_up_50_val = d_up_50_val/ d_edge_sum

    d_decode_val = d_up_50_val*bins[up_50_idx] + d_on_50_val* bins[on_50_idx]
    
    decoded_val[down_edge_mask] = d_decode_val[down_edge_mask]
    
    # Calculate std
    pred_std = None
    if whether_std == True:
        pred_std = decoder_std(n_prob, decoded_val, bins)
    
    if whether_CI == False:
        return decoded_val, None, None, pred_std
    else:
        l_cumu_prob = torch.cumsum(n_prob, dim=0)
        h_cumu_prob = l_cumu_prob.clone()

        # Lower first
        l_cumu_prob[l_cumu_prob < 0.05] = 999 
        # l_lw_idx over 0.05 
        trash, l_lw_idx = torch.min(l_cumu_prob, dim=0)
        
        l_lw_down_idx = l_lw_idx.clone()
        l_lw_down_idx -= 1 
        l_lw_edge_mask = (l_lw_down_idx == -1)
        l_lw_down_idx[l_lw_edge_mask] = 2 

        # Get the probs
        l_cumu_n_prob = torch.cumsum(n_prob, dim=0)
        l_lw_prob = l_cumu_n_prob[l_lw_idx, torch.arange(H)[:, None], torch.arange(W)]
        l_lw_down_prob = l_cumu_n_prob[l_lw_down_idx, torch.arange(H)[:, None], torch.arange(W)]
        l_alpha = (l_lw_prob - 0.05)/(l_lw_prob - l_lw_down_prob)
        
        lw_ret = bins[l_lw_idx] - l_alpha*(bins[l_lw_idx] - bins[l_lw_down_idx])
        
        lw_ret[l_lw_edge_mask] = (bins[l_lw_idx])[l_lw_edge_mask]
        
        
        # Upper
        h_cumu_prob[h_cumu_prob < 0.95] = 999 
        # l_lw_idx over 0.05 
        trash, h_lw_idx = torch.min(h_cumu_prob, dim=0)
        
        h_lw_down_idx = h_lw_idx.clone()
        h_lw_down_idx -= 1 
        h_lw_edge_mask = (h_lw_down_idx == -1)
        h_lw_down_idx[h_lw_edge_mask] = 2 

        # Get the probs
        h_cumu_n_prob = torch.cumsum(n_prob, dim=0)
        h_lw_prob = h_cumu_n_prob[h_lw_idx, torch.arange(H)[:, None], torch.arange(W)]
        h_lw_down_prob = h_cumu_n_prob[h_lw_down_idx, torch.arange(H)[:, None], torch.arange(W)]
        h_alpha = (h_lw_prob - 0.95)/(h_lw_prob - h_lw_down_prob)
        
        hw_ret = bins[h_lw_idx] - h_alpha*(bins[h_lw_idx] - bins[h_lw_down_idx])
        hw_ret[h_lw_edge_mask] = (bins[h_lw_idx])[h_lw_edge_mask]
        return decoded_val, lw_ret, hw_ret, pred_std



#   Modified for Azimuth
def decoder_median_with_noise_Azimuth(prob:torch.tensor, bins:torch.tensor, whether_CI = False,
                                      whether_std = False, reproducible = False, fixed_noise = None,
                                      hashing_key = None):
    ########
    # Args:
    #
    #   prob: Tensor of probability (N,K,H,W)
    #   bins: Tensor of bins (Kx1)
    #   
    #
    # Return:
    #   value: Tensor of values (N,C,H,W) N=C=1
    ########
    assert(torch.is_tensor(prob) and torch.is_tensor(bins))
    GLOBAL_DEVICE = prob.device
    bins = bins.to(GLOBAL_DEVICE)
    
    # First convert value to #bins x H x W
    orig_prob = torch.squeeze(prob).float()
    
    triangle_noise = triangular_noise_generation_packer_through_torch_seed(orig_prob.shape, reproducible, hashing_key, device=GLOBAL_DEVICE)
    
    n_prob = orig_prob.clone() + triangle_noise
    n_prob = torch.softmax(n_prob.clone(), dim=0)
    
    # c_prob is equal to n_prob now
    c_prob = n_prob.clone()
    
    assert(len(c_prob.size()) == 3)
    assert(len(bins.size()) == 1)
    K, = bins.size()
    C,H,W = c_prob.size()
    assert(C == K)
    
    # Find the mean bin, first let all <0.5 prob to be very large and find the min
    trash, on_50_idx = torch.max(c_prob, dim=0)
    
    down_50_idx = on_50_idx.clone()
    up_50_idx = on_50_idx.clone()
    
    # down_50_idx is just over_50_idx - 1
    # Mind -1, take -1 to 1
    down_50_idx -= 1 
    down_edge_mask = (down_50_idx == -1) 
    down_50_idx[down_edge_mask] = 0

    up_50_idx += 1
    up_edge_mask = (up_50_idx >= K)
    up_50_idx[up_edge_mask] = 0 


    u_on_50_val = n_prob[on_50_idx, torch.arange(H)[:, None], torch.arange(W)]
    u_down_50_val = n_prob[down_50_idx, torch.arange(H)[:, None], torch.arange(W)]
    

    d_on_50_val = n_prob[on_50_idx, torch.arange(H)[:, None], torch.arange(W)]
    d_up_50_val = n_prob[up_50_idx, torch.arange(H)[:, None], torch.arange(W)]
 
   
    o_on_50_val = n_prob[on_50_idx, torch.arange(H)[:, None], torch.arange(W)]
    o_down_50_val = n_prob[down_50_idx, torch.arange(H)[:, None], torch.arange(W)]
    o_up_50_val = n_prob[up_50_idx, torch.arange(H)[:, None], torch.arange(W)]

    # Calculate total sum
    val_sum = o_on_50_val + o_down_50_val + o_up_50_val


    o_down_50_val = o_down_50_val/val_sum
    o_on_50_val = o_on_50_val/val_sum
    o_up_50_val = o_up_50_val/val_sum 
    

    decoded_val = o_down_50_val* bins[down_50_idx] + o_on_50_val* bins[on_50_idx] + o_up_50_val* bins[up_50_idx]
    
    
    # Deal with edge cases by overwrite according to masks

    # Reach upper limit
    u_edge_sum = u_on_50_val + u_down_50_val
    u_on_50_val = u_on_50_val/u_edge_sum
    u_down_50_val = u_down_50_val/ u_edge_sum

    u_decode_val = u_down_50_val*bins[down_50_idx] + u_on_50_val* bins[on_50_idx]
    
    decoded_val[up_edge_mask] = u_decode_val[up_edge_mask]


    # Reach lower limit
    d_edge_sum = d_on_50_val + d_up_50_val
    d_on_50_val = d_on_50_val/d_edge_sum
    d_up_50_val = d_up_50_val/ d_edge_sum

    d_decode_val = d_up_50_val*bins[up_50_idx] + d_on_50_val* bins[on_50_idx]
    
    decoded_val[down_edge_mask] = d_decode_val[down_edge_mask]
    
    # Calculate std
    pred_std = None
    if whether_std == True:
        pred_std = decoder_std(n_prob, decoded_val, bins)
    
    if whether_CI == False:
        
        return decoded_val, None, None, pred_std
    else:
        # total_prob: [n_prob, n_prob, n_prob]
        total_prob = (n_prob.clone()).repeat((3,1,1))
        trash, center_idx = torch.max(n_prob, dim=0)

        t_cumu_prob = torch.cumsum(total_prob, dim=0)
        t_center_idx = center_idx + K 
        t_center_prob = t_cumu_prob[t_center_idx, torch.arange(H)[:, None], torch.arange(W)]

        t_low_prob = t_center_prob.clone() - 0.45
        t_high_prob = t_center_prob.clone() + 0.45 

        
        l_cumu_prob = torch.cumsum(total_prob, dim=0)
        h_cumu_prob = l_cumu_prob.clone()
        

        # Lower first
        l_cumu_prob[l_cumu_prob < t_low_prob] = 999
        trash, l_lw_idx = torch.min(l_cumu_prob, dim=0)

        l_lw_down_idx = l_lw_idx.clone()
        l_lw_down_idx -= 1 
        
        # Edge case: Fall in 180, 0
        l_lw_edge_mask = (l_lw_idx == K)
        

        # Get the probs
        l_cumu_n_prob = torch.cumsum(total_prob, dim=0)
        l_lw_prob = l_cumu_n_prob[l_lw_idx, torch.arange(H)[:, None], torch.arange(W)]
        l_lw_down_prob = l_cumu_n_prob[l_lw_down_idx, torch.arange(H)[:, None], torch.arange(W)]
        l_alpha = (l_lw_prob - t_low_prob)/(l_lw_prob - l_lw_down_prob)
        
        l_lw_idx = l_lw_idx % K 
        l_lw_down_idx = l_lw_down_idx % K 
        lw_ret = bins[l_lw_idx] - l_alpha*(bins[l_lw_idx] - bins[l_lw_down_idx])
        lw_ret[l_lw_edge_mask] = (bins[l_lw_idx])[l_lw_edge_mask]
        
        
        # Upper
        h_cumu_prob[h_cumu_prob < t_high_prob] = 999 
        # l_lw_idx over 0.05 
        trash, h_lw_idx = torch.min(h_cumu_prob, dim=0)
        h_lw_down_idx = h_lw_idx.clone()
        h_lw_down_idx -= 1 



        # Edge case: Fall in 180, 0 between the first/second ; second/third  
        h_lw_edge_mask = torch.logical_or((h_lw_down_idx == K), (h_lw_down_idx == 2*K))
        
        # Get the probs
        h_cumu_n_prob = torch.cumsum(total_prob, dim=0)
        h_lw_prob = h_cumu_n_prob[h_lw_idx, torch.arange(H)[:, None], torch.arange(W)]
        h_lw_down_prob = h_cumu_n_prob[h_lw_down_idx, torch.arange(H)[:, None], torch.arange(W)]
        h_alpha = (h_lw_prob - t_high_prob)/(h_lw_prob - h_lw_down_prob)
        

        h_lw_idx = h_lw_idx % K 
        h_lw_down_idx = h_lw_down_idx % K 
        hw_ret = bins[h_lw_idx] - h_alpha*(bins[h_lw_idx] - bins[h_lw_down_idx])
        hw_ret[h_lw_edge_mask] = (bins[h_lw_idx])[h_lw_edge_mask]

        return decoded_val, lw_ret, hw_ret, pred_std


