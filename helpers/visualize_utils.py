import matplotlib.pyplot as plt
import numpy as np
import os
import astropy.io.fits as fits
import concurrent.futures



_visField = ('plasma',0,5000)
_visIncl = ('Spectral',0,180)
_visAzim = ('twilight',0,180)
#_visBHelio = ('PuOr',-3000,3000)
_visBHelio = ('PuOr',-1000,1000)
_visBHelio_Pole = ('PuOr',-250,250)

_visKey = {'spInv_aB': _visField, 'spInv_Field_Strength': _visField, 
        'spInv_Field_Inclination': _visIncl, 'spInv_Field_Azimuth':_visAzim,
        'spInv_Stray_Light_Fill_Factor': ('Purples',0,1),
        'spDisambig_Field_Azimuth_Disamb': ('hsv',-180,180),
        'spDisambig_Bp': _visBHelio, 'spDisambig_Bt': _visBHelio, 'spDisambig_Br': _visBHelio,
        'spDisambig_Bp_Pole': _visBHelio_Pole, 'spDisambig_Bt_Pole': _visBHelio_Pole, 'spDisambig_Br_Pole': _visBHelio_Pole,
        'field': _visField, 'inclination': _visIncl, 'azimuth': _visAzim,
        'vlos_mag':("coolwarm",-700000,700000),
        'dop_width':("viridis",0,60),
        "eta_0":("cividis",0,60),
        "src_continuum":("Reds",0,29000),
        "src_grad":("Oranges",0,52000),
        "Doppler":("coolwarm", -6, 6)}


def get_hexbin_range(key):
    cmap_val, vmin_val, vmax_val = _visKey[key]
        
    if key == 'spDisambig_Br' or key == 'spDisambig_Bp' or key == 'spDisambig_Bt':
        vmin_val = -3000**0.5
        vmax_val = 3000**0.5
    return cmap_val, vmin_val, vmax_val


def signed_sqrt(x):
    return np.sign(x) * np.sqrt(np.abs(x))

#   Visualize each fits file in the directory
#   Note: for Br/Bp/Bt, it will by default visualize them in signed sqrt scale for better contrast
def process_vis_file(i, pred_file_list, err_file_list, dir_path, save_path, y_name):
    file_path = os.path.join(dir_path, pred_file_list[i])
    arr = fits.open(file_path)
    arr = arr[1].data
    
    # !!! Flip arr
    #arr = np.flip(arr, axis=1)

    img_name = pred_file_list[i].replace('.fits', '.png')
    if 'Br' in img_name or 'Bp' in img_name or 'Bt' in img_name:
        arr = signed_sqrt(arr)
        #print(f'{img_name}')
    cmap_val, vmin_val, vmax_val = get_hexbin_range(y_name)
    plt.imsave(os.path.join(save_path, img_name), arr, cmap=cmap_val, vmin=vmin_val, vmax=vmax_val)

    
    
    if err_file_list:
        file_path = os.path.join(dir_path, err_file_list[i])
        arr = fits.open(file_path)
        arr = arr[1].data

        img_name = pred_file_list[i].replace('.fits', '.uncertainty.png')
        plt.imsave(os.path.join(save_path, img_name), arr, cmap='hot', vmin=0, vmax=2000)


def fits_vis_packer(output_name_list, dir_base_path, save_base_path, every_n = 1):
    for y_name in output_name_list:
        dir_path = os.path.join(dir_base_path, y_name)
        save_path = os.path.join(save_base_path, y_name)

        # Get a list of all npy files in the directory
        pred_file_list = [file for file in sorted(os.listdir(dir_path)) if file.endswith('.fits') and 'err' not in file]
        err_file_list = [file for file in sorted(os.listdir(dir_path)) if file.endswith('_err.fits')]
        os.makedirs(save_path, exist_ok=True)
        # Delete all files in the directory
        for file in os.listdir(save_path):
            file_path = os.path.join(save_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                
        # Visualize every nth npy file
        # Check if running on Slurm
        max_workers = 1
        if 'SLURM_CPUS_PER_TASK' in os.environ:
            # Get the number of CPUs allocated by Slurm
            max_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
        else:
            # Fall back to the number of CPUs on the local machine
            max_workers = os.cpu_count()
        with concurrent.futures.ProcessPoolExecutor(max_workers = max_workers) as executor:
            n = every_n
            executor.map(process_vis_file, range(0, len(pred_file_list), n), [pred_file_list]*len(range(0, len(pred_file_list), n)),
                        [err_file_list]*len(range(0, len(pred_file_list), n)), [dir_path]*len(range(0, len(pred_file_list), n)),
                        [save_path]*len(range(0, len(pred_file_list), n)), [y_name]*len(range(0, len(pred_file_list), n)))
