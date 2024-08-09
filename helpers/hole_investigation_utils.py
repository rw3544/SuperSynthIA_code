import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.special import softmax



def extract_timestamp(file_name):
    parts = file_name.split('.')
    date_time_str = parts[2]  # This will be '20170712_070000_TAI'
    timestamp = date_time_str[:-4]  # Removes the '_TAI' part
    return timestamp

# Packer that read in three npys, and plot row of subplots accordingly
#  flag: 'large'/'masked/
def Hole_Checking_Vis_Packer(NPY_SAVE_DIR, VIS_DIR, flag, num_sampled = 2000):
    iquv_arr = np.load(os.path.join(NPY_SAVE_DIR, f'{flag}_iquv.npy'))
    pred_arr = np.load(os.path.join(NPY_SAVE_DIR, f'{flag}_pred.npy'))
    logit_arr = np.load(os.path.join(NPY_SAVE_DIR, f'{flag}_logit.npy'))
    filename_arr = np.load(os.path.join(NPY_SAVE_DIR, f'{flag}_filename.npy'))
    _, N = iquv_arr.shape
    
    if not os.path.exists(VIS_DIR):
        os.makedirs(VIS_DIR)
    # Determine the indices to process
    if N < num_sampled:
        indices = np.arange(N)
    else:
        indices = np.random.choice(N, num_sampled, replace=False)
        np.random.shuffle(indices)
    
    
    for i in indices:
        iquv = iquv_arr[:, i]
        pred = pred_arr[i]
        logit = logit_arr[:, i]
        filename = filename_arr[i]
        
        vis_samp(iquv, pred, logit, filename, VIS_DIR, i)
    

# Vmin/Vmax for all the subplots: 
#   I: 2000, 30000(max)
#   Q: -5000, 5000
#   U: -5000, 5000
#   V: -5000, 5000
#   logit: -120, 0

def vis_samp(iquv, pred, logit, filename, VIS_DIR, i):
    I = iquv[:6]
    Q = iquv[6:12]
    U = iquv[12:18]
    V = iquv[18:24]
    fig, axs = plt.subplots(1, 7, figsize=(35, 5))
    
    axs[0].plot(I)
    axs[0].set_title('I')
    axs[0].set_ylim(2000, 30000)
    
    axs[1].plot(Q)
    axs[1].set_title('Q')
    axs[1].set_ylim(-5000, 5000)
    
    axs[2].plot(U)
    axs[2].set_title('U')
    axs[2].set_ylim(-5000, 5000)
    
    axs[3].plot(V)
    axs[3].set_title('V')
    axs[3].set_ylim(-5000, 5000)
    
    axs[4].plot(logit)
    axs[4].set_title('logit')
    axs[4].set_ylim(-120, 0)
    
    pdf = softmax(logit)
    axs[5].plot(pdf)
    axs[5].set_title('pdf')
    axs[5].set_ylim(0, 1)
    
    # pdf and cdf through softmax
    cdf = np.cumsum(pdf)
    axs[6].plot(cdf)
    axs[6].set_title('cdf')
    axs[6].set_ylim(0, 1)
    
    
    fig.suptitle(f'Time: {extract_timestamp(filename)}; Pred: {np.round(pred,2)}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, f'{i}_{filename}.png'))

# Visualize histogram for I
def vis_histogram(NPY_SAVE_DIR, VIS_DIR,):
    if not os.path.exists(VIS_DIR):
        os.makedirs(VIS_DIR)
    large_iquv_arr = np.load(os.path.join(NPY_SAVE_DIR, f'large_iquv.npy'))
    masked_iquv_arr = np.load(os.path.join(NPY_SAVE_DIR, f'masked_iquv.npy'))
    
    # I
    large_i = large_iquv_arr[:6]
    masked_i = masked_iquv_arr[:6]
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    
    axs[0].hist(large_i.flatten(), bins=100, density=True)
    axs[0].set_xlim(2000, 25000)
    axs[0].set_ylim(0, 0.001)
    axs[0].set_title('hist of large I')
    
    axs[1].hist(masked_i.flatten(), bins=100, density=True)
    axs[1].set_xlim(2000, 25000)
    axs[1].set_ylim(0, 0.001)
    axs[1].set_title('hist of masked I')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, f'Histogram_I.png'))
    
    
    # Q
    large_q = large_iquv_arr[6:12]
    masked_q = masked_iquv_arr[6:12]
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    
    axs[0].hist(large_q.flatten(), bins=100, density=True)
    axs[0].set_xlim(-2000, 2000)
    axs[0].set_ylim(0, 0.007)
    axs[0].set_title('hist of large Q')
    
    axs[1].hist(masked_q.flatten(), bins=100, density=True)
    axs[1].set_xlim(-2000, 2000)
    axs[1].set_ylim(0, 0.007)
    axs[1].set_title('hist of masked Q')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, f'Histogram_Q.png'))
    
    
    # U
    large_u = large_iquv_arr[12:18]
    masked_u = masked_iquv_arr[12:18]
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    
    axs[0].hist(large_u.flatten(), bins=100, density=True)
    axs[0].set_xlim(-1500, 1500)
    axs[0].set_ylim(0, 0.007)
    axs[0].set_title('hist of large U')
    
    axs[1].hist(masked_u.flatten(), bins=100, density=True)
    axs[1].set_xlim(-1500, 1500)
    axs[1].set_ylim(0, 0.007)
    axs[1].set_title('hist of masked U')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, f'Histogram_U.png'))
    
    
    # V
    large_v = large_iquv_arr[18:]
    masked_v = masked_iquv_arr[18:]
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    
    axs[0].hist(large_u.flatten(), bins=100, density=True)
    axs[0].set_xlim(-1500, 1500)
    axs[0].set_ylim(0, 0.01)
    axs[0].set_title('hist of large V')
    
    axs[1].hist(masked_u.flatten(), bins=100, density=True)
    axs[1].set_xlim(-1500, 1500)
    axs[1].set_ylim(0, 0.01)
    axs[1].set_title('hist of masked V')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, f'Histogram_V.png'))
    
    
