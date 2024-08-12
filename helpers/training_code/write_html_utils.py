import os
import numpy as np
from evaluate_utils import *

def write_html(eval_dict, IMG_DIR, TIMESTAMP_DIR, key, Pole_only_index = None, bin_class = None):
    ########
    # Args:
    # 
    #
    #
    ########
    Timestamp_arr = np.load(TIMESTAMP_DIR)
    HTML_PATH = os.path.join(IMG_DIR, 'visualize.html')
    func = open(HTML_PATH, "w")
    
    func.write(f"<html>\n<head>\n<title> \n HMI \ </title>\n</head> <body><h1> Data for {key} | Diff | True | Predict | hex </h1>")
    
    img_idx_list = np.load(os.path.join(IMG_DIR, 'idx_list.npy'))
    
    NPY_SAVE_DIR = eval_dict['NPY_SAVE_DIR']
    NUM_SAMPLES = eval_dict['NUM_SAMPLES']
    Test_y_ZARR_DIR = eval_dict['Test_y_ZARR_DIR']
    MASK_zarr_DIR = eval_dict['MASK_zarr_DIR']
    bin_class = eval_dict['bin_class']
    t = eval_dict['t']
    
    GT = zarr.open(Test_y_ZARR_DIR, mode='r')
    MASK_zarr = zarr.open(MASK_zarr_DIR, mode='r')
    
    

    total_MAE, total_RMSE, total_PGP = complete_evaluation(NPY_SAVE_DIR, NUM_SAMPLES, Test_y_ZARR_DIR, MASK_zarr_DIR, bin_class, t, key, IMG_DIR)
    
    evaluation = {"MAE": total_MAE,
                  "RMSE": total_RMSE,
                  "PGP": total_PGP}
    
    description = 'standard'
    if bin_class != None:
        description = bin_class._get_description()
    
    np.save(os.path.join(NPY_SAVE_DIR, f'eval_result_{description}.npy'), evaluation)
    
    if Pole_only_index == None:
        func.write(f"        <p> % within t = {t}:{total_PGP} </p> \n")
        func.write(f"        <p> Overall RMSE Error:{total_RMSE} </p> \n")
        func.write(f"        <p> Overall MAE Error:{total_MAE} </p> \n")
        func.write(f"        <p> Histogram: </p> \n")
        func.write(f"        <img src='{f'hexbin_{description}.png'}'>\n")
        func.write("         <hr>")
        func.write('<div class="column"> \n')
    else:
        
        print('Pole only')
        quit()
        if percentage_arr != []:
            func.write(f"        <p> % within t = {t}:{calculate_combined_mae_percentage(percentage_arr[:, Pole_only_index:])} </p> \n")
        func.write(f"        <p> Pole only MSE Error:{np.mean(mse_loss_arr[Pole_only_index:])} </p> \n")
        func.write(f"        <p> Pole only MAE Error:{np.mean(mae_loss_arr[Pole_only_index:])} </p> \n")
        func.write(f"        <p> Histogram: </p> \n")
        func.write(f"        <img src='{f'hist_Pole.png'}'>\n")
        func.write("         <hr>")
        func.write('<div class="column"> \n')
    
    for i in img_idx_list:
        timestamp = Timestamp_arr[i][0]
        

        # Calculate per image MAE, RMSE, PGP
        PRED_DIR = os.path.join(NPY_SAVE_DIR, f'{i}_predict_{timestamp}.npy')
        TRUE_arr = load_y(GT, i)
        pred_arr = np.load(PRED_DIR)
        pred_arr = pred_arr.squeeze()
        dmask = load_eroded_mask(MASK_zarr, i)
            
        y_mask = ~np.isnan(TRUE_arr)
        dmask *= y_mask
            
        MAE, RMSE, PGP = per_img_MSE_MAE_percentage_loss(pred_arr, TRUE_arr, dmask, bin_class, t)
        
        
  
        
        func.write('    <div class="row">\n')
        #DIFF_PATH = os.path.join(MAIN_PATH, f'{i}_diff.png')
        #TRUE_PATH = os.path.join(MAIN_PATH, f'{i}_true.png')
        #PRED_PATH = os.path.join(MAIN_PATH, f'{i}_predict.png')
        
        if bin_class == None:    
            func.write(f"        <img src='{f'{i}_{timestamp}_diff.png'}'>\n")
            func.write(f"        <img src='{f'{i}_{timestamp}_true.png'}'>\n")
            func.write(f"        <img src='{f'{i}_{timestamp}_predict.png'}'>\n")
            func.write(f"        <img src='{f'{i}_{timestamp}_hex.png'}'>")
        else:
            func.write(f"        <img src='{f'{i}_{timestamp}_{bin_class._get_description()}_diff.png'}'>\n")
            func.write(f"        <img src='{f'{i}_{timestamp}_{bin_class._get_description()}_true.png'}'>\n")
            func.write(f"        <img src='{f'{i}_{timestamp}_{bin_class._get_description()}_predict.png'}'>\n")
            func.write(f"        <img src='{f'{i}_{timestamp}_{bin_class._get_description()}_hex.png'}'>")
            
        func.write(f"        <p> index={i}, timestamp={Timestamp_arr[i][0]},  RMSEloss={RMSE}, MAEloss={MAE} </p> \n")
        func.write(f"        <p> % within t:{PGP} </p> \n")
        func.write(f"        <hr>")
        func.write('</div>\n')
        if i %10 == 0:
            func.write("<style> @media print{h1 {page-break-before:always}}</style>")
    
    func.write('</div>\n')
    

    func.write("\n</body></html>")


