from cmath import nan
import numpy as np
import os
import zarr
import random
from numcodecs import Blosc

from utils import verify_zarr, de_pad

# consider cupy to speed up numpy

GLOBAL_PADDING_SIZE = 1024

# Pad to predefined size

########
# int(padded[-1,0]) = original_shape[0]
# int(padded[-1,1]) = original_shape[1]
# padded[-1,0,0] if the array is 3d
########

def pre_pad(
    x,
    ):
    x_h = x.shape[0]
    x_w = x.shape[1]

    if x_h >= GLOBAL_PADDING_SIZE or x_w >= GLOBAL_PADDING_SIZE:
        raise ValueError(f'Original Image {x_h}, {x_w} greater than defined Padding size')
    
    pad_h = GLOBAL_PADDING_SIZE - x_h
    pad_w = GLOBAL_PADDING_SIZE - x_w + 1

    #print(f'Original: {x.shape}')

    if len(x.shape) == 2:
        ret = np.pad(x,((0,pad_h),(0,pad_w)), 'constant')
        ret[-1,0] = x_h
        ret[-1,1] = x_w
    elif len(x.shape) == 3:
        ret = np.pad(x,((0,pad_h),(0,pad_w),(0,0)), 'constant')
        ret[-1,0,0] = x_h
        ret[-1,1,0] = x_w
    else:
        raise ValueError('Unsupported Dimension for pre_pad function')

    

    #print(f'Padded: {ret.shape}')

    return ret


def year_list_to_str(year_list):
    store = ''
    for year in year_list:
        store += f'{year}_'
    return store

# Make sure the c_year is in the right format
def verify_c_year(c_year:str):
    assert(len(c_year) == 7)
    assert(int(c_year[-1]) == 1 or int(c_year[-1]) == 2)

def preprocess_Data(
    srcBase: str,
    srcFlowBase: str,
    srcFlowDisambigBase: str,
    targetBase: str,
    c_year: str,
    DEBUG_MODE = False,
    QUIET_MODE = False,
    ZARR_VERIFY = True
    ):
    ########
    # Important Args:
    # c_year: The chosen (half) year for training dataset
    #         2011_01: 1-6
    #         2011_02: 7-12
    # 
    # Directory:
    #   target_dir: The base target directory for saved zarrs
    #   
    ########
    # the year chosen to be input
    target_dir = targetBase    
    
    
    verify_c_year(c_year)


    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    if QUIET_MODE == False:
        print(f'Loading data for year: {c_year}')
        print()

    # list of name of all folder that starts with c_year
    #Base_folder_list = [foldername for foldername in sorted(os.listdir(srcBase)) if
    #    foldername.startswith(c_year)]
    # !!! Folders in base and flowbase has the same name
    
    FlowComb_folder_list = None
    if int(c_year[-1]) == 1:
        FlowBase_folder_list = [foldername for foldername in sorted(os.listdir(srcFlowBase)) if
            foldername.startswith(c_year[:4]) and int(foldername[4:6])<=6]
        FlowDisambig_folder_list = [foldername for foldername in sorted(os.listdir(srcFlowDisambigBase)) if
            foldername.startswith(c_year[:4]) and int(foldername[4:6])<=6]   
        # Take the intersection of these two list
        FlowComb_folder_list = list(set(FlowBase_folder_list).intersection(set(FlowDisambig_folder_list))) 
    if int(c_year[-1]) == 2:
        FlowBase_folder_list = [foldername for foldername in sorted(os.listdir(srcFlowBase)) if
            foldername.startswith(c_year[:4]) and int(foldername[4:6])>6]
        FlowDisambig_folder_list = [foldername for foldername in sorted(os.listdir(srcFlowDisambigBase)) if
            foldername.startswith(c_year[:4]) and int(foldername[4:6])>6]
        FlowComb_folder_list = list(set(FlowBase_folder_list).intersection(set(FlowDisambig_folder_list))) 
    FlowComb_folder_list = sorted(FlowComb_folder_list)
    FlowBase_folder_list = None
    
    
    key = 'DISK'
    if 'Pole' in srcBase:
        key = 'Pole'
        assert('Pole' in srcFlowBase)
        assert('Pole' in srcFlowDisambigBase)
    else:
        assert('Pole' not in srcFlowBase)
        assert('Pole' not in srcFlowDisambigBase)
        

    # The length of all zarr
    Global_length_zarr = 0
    # Get the correct length of zarr (Exclude those with no npy)
    for i_main in range(len(FlowComb_folder_list)):
        foldername = FlowComb_folder_list[i_main]
        # use src to access file in folder for each measurement
        # src: srcBase and flowBase_folder list
        src = os.path.join(srcBase, foldername)
        
        # For all scan with same .fits folder
        for obsDate in sorted(os.listdir(src)):
            if obsDate.find("_") == -1 or not os.path.isdir(os.path.join(src,obsDate)):
                continue
            volFile = os.path.join(src, obsDate, obsDate+"_hmiObs.npz")
            #volInvFile = os.path.join(src, obsDate, obsDate+"_hmiInv.npz")
            volSPInvFile = os.path.join(srcFlowBase, foldername, obsDate, obsDate+"_flowedSPInv.npz")
            volSPDisambigFile = os.path.join(srcFlowDisambigBase, foldername, obsDate, obsDate+"_flowedSPDisambig.npz")
            
            
            # check if file exists, if not, ignore this folder
            if os.path.exists(volFile) == False:
                ERR_MESSAGE_DIR = os.path.join(target_dir, f'ERROR_{c_year}_{key}.txt')
                with open(ERR_MESSAGE_DIR, 'a') as f:
                    f.write(volFile)
                    f.write('\n')
                continue
            
            if os.path.exists(volSPInvFile) == False:
                ERR_MESSAGE_DIR = os.path.join(target_dir, f'ERROR_{c_year}_{key}.txt')
                with open(ERR_MESSAGE_DIR, 'a') as f:
                    f.write(volSPInvFile)
                    f.write('\n')
                continue
            
            if os.path.exists(volSPDisambigFile) == False:
                ERR_MESSAGE_DIR = os.path.join(target_dir, f'ERROR_{c_year}_{key}.txt')
                with open(ERR_MESSAGE_DIR, 'a') as f:
                    f.write(volSPDisambigFile)
                    f.write('\n')
                continue
            
            Global_length_zarr += 1
            
            # Check the maximum size, DELETE FOR LATER
            if False:
                vol = np.load(volFile)
                volSPInv = np.load(volSPInvFile)
                volSPDisambig = np.load(volSPDisambigFile)
                c_vol_max = np.max(vol['hmiStokes'].shape)
                c_SP_max = np.max(volSPInv['spInversion'].shape)
                c_SPDIS_max = np.max(volSPDisambig['spInversion'].shape)
                c_max = max(c_vol_max, c_SP_max, c_SPDIS_max)
                if c_max > MAXMAXMAXSHAPE:
                    MAXMAXMAXSHAPE = c_max
                    print(MAXMAXMAXSHAPE)
    
    
    
    
    # If file already exists, ask if override
    # Will not override except enter 0
    if os.path.exists(os.path.join(target_dir, f'X_{c_year}_{key}.zarr')):
        print(f'File already exists, sure to OVERRIDE?')
        print(f' To continue, enter 0')
        tmp = input()
        if tmp != '0':
            print('No override, exit program now.')
            quit()
         
         
    # Variables to be saved 
    # Everything is saved as N,C,H,W 
    
    compressor = Blosc(cname='lz4',clevel=2,shuffle=1)
    # Hardcode 24 channels for IQUV
    ZARR_comb_stoke = zarr.open(
        os.path.join(target_dir, f'X_{c_year}_{key}.zarr'),
        mode = 'w',
        shape=(Global_length_zarr, 24, GLOBAL_PADDING_SIZE, GLOBAL_PADDING_SIZE+1),
        chunks=(1,24, GLOBAL_PADDING_SIZE, GLOBAL_PADDING_SIZE+1),
        dtype = np.float32,
        compressor=compressor
    )
    ZARR_spInv_gt_label = {}
    ZARR_disambig_gt_label = {}
    #hmiInv_gt_label = {}
    ZARR_time_error = zarr.open(
        os.path.join(target_dir, f'timeError_{c_year}_{key}.zarr'),
        mode = 'w',
        shape=(Global_length_zarr, 1, GLOBAL_PADDING_SIZE, GLOBAL_PADDING_SIZE+1),
        chunks=(1,1, GLOBAL_PADDING_SIZE, GLOBAL_PADDING_SIZE+1),
        dtype = np.float32,
        compressor=compressor
    )
    # Not N,C,H,W, just N
    ZARR_hmi_header = {}
    ZARR_hmi_metadata_gt_label = {}
    #hmi_heliographic_gt_label = {}
    
    initialized = False
    tmp_number = 0
    # First Initialize all the ZARRS in the dictionary
    while initialized == False:
        foldername = FlowComb_folder_list[tmp_number]
        src = os.path.join(srcBase, foldername)

        print(f'src: {src}')
        
        for obsDate in sorted(os.listdir(src)):
            if obsDate.find("_") == -1 or not os.path.isdir(os.path.join(src,obsDate)):
                tmp_number += 1
                continue
            volFile = os.path.join(src, obsDate, obsDate+"_hmiObs.npz")
            #volInvFile = os.path.join(src, obsDate, obsDate+"_hmiInv.npz")
            volSPInvFile = os.path.join(srcFlowBase, foldername, obsDate, obsDate+"_flowedSPInv.npz")
            volSPDisambigFile = os.path.join(srcFlowDisambigBase, foldername, obsDate, obsDate+"_flowedSPDisambig.npz")
            
            if not os.path.exists(volFile):
                tmp_number += 1
                continue
            if not os.path.exists(volSPInvFile):
                tmp_number += 1
                continue
            if not os.path.exists(volSPDisambigFile):
                tmp_number += 1
                continue
            vol = np.load(volFile)
            #volInv = np.load(volInvFile)
            volSPInv = np.load(volSPInvFile)
            volSPDisambig = np.load(volSPDisambigFile)

            # HMI headers
            for ki,k in enumerate([k for k in vol.keys() if k.startswith("HMIHEADER")]):
                # Get OBS_VR, OBS_VN, OBS_VW
                tmp_key = k.replace("HMIHEADER_","")
                if (not tmp_key in ZARR_hmi_header.keys()) and 'OBS_V' in tmp_key:
                    
                    ZARR_hmi_header[tmp_key] = zarr.open(
                        os.path.join(target_dir, f'hmiHeader_{tmp_key}_{c_year}_{key}.zarr'),
                        mode = 'w',
                        shape=(Global_length_zarr),
                        chunks=(1),
                        dtype = np.float32,
                        compressor=compressor
                    )
        

            # !!! gt_labels
            for i in range(volSPInv['spInversionNames'].size):
                V = volSPInv['spInversion'][:,:,i]
                if volSPInv['spInversionNames'][i] == 'Field_Azimuth':
                    V = np.fmod(V+90,180)
                
                # check if key exists
                tmp_key = volSPInv['spInversionNames'][i]
                if not tmp_key in ZARR_spInv_gt_label.keys():
                    ZARR_spInv_gt_label[tmp_key] = zarr.open(
                        os.path.join(target_dir, f'spInv_{tmp_key}_{c_year}_{key}.zarr'),
                        mode = 'w',
                        shape=(Global_length_zarr, 1, GLOBAL_PADDING_SIZE, GLOBAL_PADDING_SIZE+1),
                        chunks=(1, 1, GLOBAL_PADDING_SIZE, GLOBAL_PADDING_SIZE+1),
                        dtype = np.float32,
                        compressor=compressor
                    )
                
            tmp_key = None
            
            # !!! hmi Metadata gt_labels
            for i in range(vol['hmiMeta'].shape[2]):
                tmp_key = i
                if not tmp_key in ZARR_hmi_metadata_gt_label.keys():
                    ZARR_hmi_metadata_gt_label[tmp_key] = zarr.open(
                        os.path.join(target_dir, f'hmiMetadata_{tmp_key}_{c_year}_{key}.zarr'),
                        mode = 'w',
                        shape=(Global_length_zarr, 1, GLOBAL_PADDING_SIZE, GLOBAL_PADDING_SIZE+1),
                        chunks=(1, 1, GLOBAL_PADDING_SIZE, GLOBAL_PADDING_SIZE+1),
                        dtype = np.float32,
                        compressor=compressor
                    )
                
           
           
            # !!! disambig_gt_labels
            
            tmp_key = None
            for i in range(volSPDisambig['spInversionNames'].size):
                tmp_key = volSPDisambig['spInversionNames'][i]
                if not tmp_key in ZARR_disambig_gt_label.keys():
                    ZARR_disambig_gt_label[tmp_key] = zarr.open(
                        os.path.join(target_dir, f'spDisambig_{tmp_key}_{c_year}_{key}.zarr'),
                        mode = 'w',
                        shape=(Global_length_zarr, 1, GLOBAL_PADDING_SIZE, GLOBAL_PADDING_SIZE+1),
                        chunks=(1, 1, GLOBAL_PADDING_SIZE, GLOBAL_PADDING_SIZE+1),
                        dtype = np.float32,
                        compressor=compressor
                    )
                #ZARR_disambig_gt_label[tmp_key].append(pre_pad(volSPDisambig['spInversion'][:,:,i]))
            tmp_key = None
            
            
            
            initialized = True
            break
            # !!! hmi Heliographic Metadata
            # Don't Include
            #for i in range(vol['hmiMetaHeliographic'].shape[2]):
            #    tmp_key = i
            #    if not tmp_key in hmi_heliographic_gt_label.keys():
            #        hmi_heliographic_gt_label[tmp_key] = []
                
    print()      
    print('Initialization Zarr Complete')
    print()
    
    
    TIMESTAMP_list = []
    foldername = None
    
    ZARR_i = 0
    # Iterate through all and save to ZARR
    for i_main in range(len(FlowComb_folder_list)):
        foldername = FlowComb_folder_list[i_main]
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
            volSPDisambigFile = os.path.join(srcFlowDisambigBase, foldername, obsDate, obsDate+"_flowedSPDisambig.npz")
            
            
            # check if file exists, if not, ignore this folder
            if os.path.exists(volFile) == False:
                
                continue
            
            if os.path.exists(volSPInvFile) == False:
                
                continue

            if os.path.exists(volSPDisambigFile) == False:
                
                continue
            
            vol = np.load(volFile)
            #volInv = np.load(volInvFile)
            volSPInv = np.load(volSPInvFile)
            volSPDisambig = np.load(volSPDisambigFile)
            
            
            # !!!
            # May need padding!
            timeError = vol['timeErrorMap']
            
            ZARR_time_error[ZARR_i,0,:,:] = pre_pad(timeError)
            #print(ZARR_time_error[0,0,:,:])
            #print(np.array_equal(ZARR_time_error[i_main,0,:,:], pre_pad(timeError), equal_nan=True))
            

            # !!!
            for ki,k in enumerate([k for k in vol.keys() if k.startswith("HMIHEADER")]):
                # Get OBS_VR, OBS_VN, OBS_VW
                tmp_key = k.replace("HMIHEADER_","")
                if (tmp_key in ZARR_hmi_header.keys()) :
                    (ZARR_hmi_header[tmp_key])[ZARR_i] = (vol[k])
                    #print(f'{tmp_key}: {vol[k]}')
                
                
            #print(ZARR_hmi_header['OBS_VR'][0])
            #print(ZARR_hmi_header['OBS_VW'][0])
            #print(ZARR_hmi_header['OBS_VN'][0])
            
            # !!! gt_labels
            for i in range(volSPInv['spInversionNames'].size):
                V = volSPInv['spInversion'][:,:,i]
                if volSPInv['spInversionNames'][i] == 'Field_Azimuth':
                    V = np.fmod(V+90,180)
                
                tmp_key = volSPInv['spInversionNames'][i]
                
                (ZARR_spInv_gt_label[tmp_key])[ZARR_i, 0, :, :] = (pre_pad(V))
                #print(np.array_equal(ZARR_spInv_gt_label[tmp_key][i_main,0,:,:], pre_pad(V), equal_nan=True))
            tmp_key = None
            
            
            
            
            # !!! disambig_gt_labels
            for i in range(volSPDisambig['spInversionNames'].size):
                tmp_key = volSPDisambig['spInversionNames'][i]                
                V = volSPDisambig['spInversion'][:,:,i]
                (ZARR_disambig_gt_label[tmp_key])[ZARR_i, 0, :, :] = pre_pad(V)
                
            tmp_key = None
            
            
            # !!! hmi Metadata gt_labels
            for i in range(vol['hmiMeta'].shape[2]):
                tmp_key = i
                ZARR_hmi_metadata_gt_label[tmp_key][ZARR_i, 0, :, :] = (pre_pad(vol['hmiMeta'][:,:,i]))
                #print(np.array_equal(ZARR_hmi_metadata_gt_label[tmp_key][i_main,0,:,:], pre_pad(vol['hmiMeta'][:,:,i]), equal_nan=True))

            
        
            
            # !!! HMI Stokes, add continuum at the back!!!!!!!!!
            stoke = vol['hmiStokes']
            pad_stoke = np.transpose(pre_pad(stoke), (2,0,1))
            ZARR_comb_stoke[ZARR_i, :, :, :] = pad_stoke
            #print(np.array_equal(ZARR_comb_stoke[i_main, :, :, :], np.transpose(pre_pad(stoke), (2,0,1))))
            
            
            TIMESTAMP_list.append((obsDate, ZARR_i, src, srcFlowBase))
            
            
            ZARR_i += 1
            

    TIMESTAMP_list = np.array(TIMESTAMP_list)
    TIMESTAMP_PATH = os.path.join(target_dir, f'TIMESTAMP_{c_year}_{key}.npy')
    np.save(TIMESTAMP_PATH, TIMESTAMP_list)
    
            
            
            
        
    

# combine data for different years
def preprocess_generate_pipeline(
    input_base_DIR:str,
    year_range: list,
    y_name: str,
    SAVE_Base_DIR: str,
    VALIDATION = True,
    IGNORE_X = True
    ):
    # Get the correct shape of the output
    T_length = 0
    
    X_shape = None
    y_shape = None
    
    source_label = None
    if 'DISK' in input_base_DIR:
        source_label = 'DISK'
    elif 'POLAR' in input_base_DIR:
        source_label = 'Pole'
    for i in range(len(year_range)):
        c_year = year_range[i]
        tmp_X_DIR = os.path.join(input_base_DIR, f'X_{c_year}_{source_label}.zarr')
        tmp_y_DIR = os.path.join(input_base_DIR, f'{y_name}_{c_year}_{source_label}.zarr')
        tmp_X_zarr = zarr.open(tmp_X_DIR, mode='r')
        tmp_y_zarr = zarr.open(tmp_y_DIR, mode='r')
        T_length += tmp_X_zarr.shape[0]
        
        X_shape = tmp_X_zarr.shape 
        y_shape = tmp_y_zarr.shape
    X_shape = (T_length, X_shape[1], X_shape[2], X_shape[3])
    y_chunk_shape = None
    if len(y_shape) == 1:
        y_shape = (T_length,)
        y_chunk_shape = (1,)
    else:
        y_shape = (T_length, y_shape[1], y_shape[2], y_shape[3])
        y_chunk_shape = (1, y_shape[1], y_shape[2], y_shape[3])

    # handle save dir
    SAVE_FOLDER_DIR = SAVE_Base_DIR
    if not os.path.exists(SAVE_FOLDER_DIR):
        os.makedirs(SAVE_FOLDER_DIR)
    TARGET_X_DIR = os.path.join(SAVE_FOLDER_DIR, f'X_pipeline_year_{year_list_to_str(year_range)}_{source_label}.zarr')
    TARGET_y_DIR = os.path.join(SAVE_FOLDER_DIR, f'{y_name}_year_{year_list_to_str(year_range)}_{source_label}.zarr')
    #print(X_shape)
    #print(y_shape)
    #print(TARGET_X_DIR)
    #print(TARGET_y_DIR)
    #print((1,X_shape[1], X_shape[2], X_shape[3]))
    

    # Use float64 so that the data is not compressed
    compressor = Blosc(cname='lz4',clevel=2,shuffle=1)
    X = None
    if IGNORE_X == False:
        X = zarr.open(
            TARGET_X_DIR,
            mode = 'w',
            shape=X_shape,
            chunks=(1,X_shape[1], X_shape[2], X_shape[3]),
            dtype = np.float32,
            compressor=compressor
        )
    y = zarr.open(
        TARGET_y_DIR,
        mode = 'w',
        shape=y_shape,
        chunks=y_chunk_shape,
        dtype = np.float32,
        compressor=compressor
    )

    if IGNORE_X == False:
        print(f'Shape of X: {X.shape}')
    print(f'Shape of y: {y.shape}')

    prev_end = 0
    total_end = 0


    TIMESTAMP_PATH = os.path.join(SAVE_FOLDER_DIR, f'TIMESTAMP_{year_list_to_str(year_range)}_{source_label}.npy')
    timestamp_list = []
    # Write the zarrs and timestamp
    for i in range(len(year_range)):
        c_year = year_range[i]
        tmp_X_DIR = os.path.join(input_base_DIR, f'X_{c_year}_{source_label}.zarr')
        tmp_y_DIR = os.path.join(input_base_DIR, f'{y_name}_{c_year}_{source_label}.zarr')
        
        tmp_timestamp_DIR = os.path.join(input_base_DIR, f'TIMESTAMP_{c_year}_{source_label}.npy')
        tmp_timestamp_arr = np.load(tmp_timestamp_DIR)        
        timestamp_list.append(tmp_timestamp_arr)
        
        print(tmp_X_DIR)
        tmp_X_zarr = zarr.open(tmp_X_DIR, mode='r')
        print(tmp_y_DIR)
        tmp_y_zarr = zarr.open(tmp_y_DIR, mode='r')
        
        assert(tmp_X_zarr.shape[0] == tmp_y_zarr.shape[0])
        
        total_end += tmp_y_zarr.shape[0]
        print(f'{prev_end}, {total_end}\n')
        if IGNORE_X == False:
            X[prev_end:total_end] = tmp_X_zarr
        y[prev_end:total_end] = tmp_y_zarr
        prev_end = total_end
    timestamp_comb = np.concatenate(timestamp_list, axis=0)
    np.save(TIMESTAMP_PATH, timestamp_comb)
    assert(total_end == T_length)
    print(f'{total_end}, {T_length}')



# Not for hmiHeader
# Save to npy of [Total #, mean, mean_of_sq E(X^2)]
def calculate_mean_4d(
    DIR:str,
    name:str,
    SAVE_DIR:str,
    ):
    print(DIR)
    # N,C,H,W
    file1 = zarr.open(DIR, mode='r')
    
    count = 0
    # Get total number of values
    Total_number = np.longdouble(0.)
    for i in range(file1.shape[0]):
        
        X = de_pad(np.transpose(file1[i],(1,2,0)))
        if X.shape == (0,0,1):
            count += 1
            print(file1)
            print(i)
        Total_number += np.sum(~np.isnan(X))
    if count != 0:
        print(f'{name}, {count}')
    
    print(f'Total number: {Total_number}')
    
    mean = np.longdouble(0.) # E(X)
    mean_sq = np.longdouble(0.) # E(X^2)
    for i in range(file1.shape[0]):
        X = de_pad(np.transpose(file1[i],(1,2,0)))
        X[X != X] = 0.0 # remove nans
        mean += np.sum(X)/Total_number
        mean_sq += np.sum(X**2)/Total_number
    t_DIR = os.path.join(SAVE_DIR, f'{name}_mean_meansq.npy')
    print(np.array([Total_number, mean, mean_sq], dtype=np.longdouble))
    np.save(t_DIR, np.array([Total_number, mean, mean_sq], dtype=np.longdouble))


# Save as arr[0,:] is mean for each channel
#         arr[1,:] is std for each channel
def save_normalization_parameter(
    Train_DIR:str, 
    Val_DIR:str,
    SAVE_DIR:str,
    name: str,
    SAMPLE_SIZE = 200,
    ):
    
    # Check if the directory is correct
    assert('TRAIN' in Train_DIR)
    assert('TEST' not in Train_DIR)
    assert('VAL' not in Train_DIR)
    assert('TRAIN' not in Val_DIR)
    assert('VAL' in Val_DIR)
    assert('TEST' not in Val_DIR)
    
    
    assert(SAMPLE_SIZE % 4 == 0)
    
    print(Train_DIR)
    
    Train_file = zarr.open(Train_DIR, mode='r')
    Val_file = zarr.open(Val_DIR, mode='r')
    
    Train_sample_size = int(SAMPLE_SIZE*3/4)
    Val_sample_size = int(SAMPLE_SIZE/4)

    
    Train_index = random.sample(range(Train_file.shape[0]), Train_sample_size)
    Val_index = random.sample(range(Val_file.shape[0]), Val_sample_size)
    
    # N,C,H,W
    #combined = np.zeros((SAMPLE_SIZE, Train_file.shape[1], Train_file.shape[2], Train_file.shape[3])
    #                    , dtype = np.float32)
    Total_number = 0
    
    t_sum = np.zeros((Train_file.shape[1]), dtype=np.longdouble) # E(X)
    
    t_sum_sq = np.zeros_like(t_sum) # E(X^2)
    
    for i in Train_index:
        X = de_pad(np.transpose(Train_file[i],(1,2,0)))
        # After pad: N,H,C
        Total_number += X.shape[0] * X.shape[1]
        X[X != X] = 0.0 # remove nans
        t_sum += np.sum(X, axis=(0,1))
        t_sum_sq += np.sum(X**2, axis=(0,1))
        
        
    
    for i in Val_index:
        X = de_pad(np.transpose(Val_file[i],(1,2,0)))
        Total_number += X.shape[0] * X.shape[1]
        X[X != X] = 0.0 # remove nans
        t_sum += np.sum(X, axis=(0,1))
        t_sum_sq += np.sum(X**2, axis=(0,1))
    #combined = np.concatenate(combined, axis=1)
    t_mean = t_sum/Total_number
    t_std = (t_sum_sq/Total_number-t_mean**2)**(1/2)
    
    save_total = np.stack((t_mean, t_std))
    assert(np.array_equal(save_total[0,:], t_mean, equal_nan=True))
    assert(np.array_equal(save_total[1,:], t_std, equal_nan=True))
    
    SAVE_PATH = os.path.join(SAVE_DIR, f'{name}_Norm_param.npy')
    np.save(SAVE_PATH, save_total)


# Save as arr[0,:] is mean for each channel
#         arr[1,:] is std for each channel
def save_normalization_parameter_OBS_V(
    Train_DIR:str, 
    Val_DIR:str,
    SAVE_DIR:str,
    name: str,
    SAMPLE_SIZE = 200,
    ):
    
    # Check if the directory is correct
    assert('TRAIN' in Train_DIR)
    assert('TEST' not in Train_DIR)
    assert('VAL' not in Train_DIR)
    assert('TRAIN' not in Val_DIR)
    assert('VAL' in Val_DIR)
    assert('TEST' not in Val_DIR)
    
    
    assert(SAMPLE_SIZE % 4 == 0)
    
    print(Train_DIR)
    
    Train_file = zarr.open(Train_DIR, mode='r')
    Val_file = zarr.open(Val_DIR, mode='r')
    
    Train_sample_size = int(SAMPLE_SIZE*3/4)
    Val_sample_size = int(SAMPLE_SIZE/4)

    
    Train_index = random.sample(range(Train_file.shape[0]), Train_sample_size)
    Val_index = random.sample(range(Val_file.shape[0]), Val_sample_size)
    
    # N,C,H,W
    #combined = np.zeros((SAMPLE_SIZE, Train_file.shape[1], Train_file.shape[2], Train_file.shape[3])
    #                    , dtype = np.float32)
    
    t_sum = np.zeros((1), dtype=np.longdouble) # E(X)
    
    t_sum_sq = np.zeros_like(t_sum) # E(X^2)
    
    for i in Train_index:
        X = Train_file[i]
        # After pad: N,H,C
        assert(X != nan)
        t_sum += X
        t_sum_sq += X**2
        
        
    
    for i in Val_index:
        X = Val_file[i]
        assert(X != nan)
        t_sum += X
        t_sum_sq += X**2
    #combined = np.concatenate(combined, axis=1)
    t_mean = t_sum/SAMPLE_SIZE
    t_std = (t_sum_sq/SAMPLE_SIZE-t_mean**2)**(1/2)
    
    save_total = np.stack((t_mean, t_std))
    assert(np.array_equal(save_total[0,:], t_mean, equal_nan=True))
    assert(np.array_equal(save_total[1,:], t_std, equal_nan=True))
    

    SAVE_PATH = os.path.join(SAVE_DIR, f'{name}_Norm_param.npy')
    np.save(SAVE_PATH, save_total)




# combine data for different years
def preprocess_generate_extra_layer(
    input_base_DIR:str,
    year_range: list,
    y_name: str,
    OBS_Name: list,
    SAVE_Base_DIR: str,
    VALIDATION = True,
    IGNORE_X = True
    ):
    # Get the correct shape of the output
    T_length = 0
    
    X_shape = None
    y_shape = None
    
    source_label = None
    if 'DISK' in input_base_DIR:
        source_label = 'DISK'
    elif 'POLAR' in input_base_DIR:
        source_label = 'Pole'
    for i in range(len(year_range)):
        c_year = year_range[i]
        tmp_X_DIR = os.path.join(input_base_DIR, f'X_{c_year}_{source_label}.zarr')
        tmp_y_DIR = os.path.join(input_base_DIR, f'{y_name}_{c_year}_{source_label}.zarr')
        tmp_X_zarr = zarr.open(tmp_X_DIR, mode='r')
        tmp_y_zarr = zarr.open(tmp_y_DIR, mode='r')
        T_length += tmp_X_zarr.shape[0]
        
        X_shape = tmp_X_zarr.shape 
        y_shape = tmp_y_zarr.shape
    X_shape = (T_length, X_shape[1], X_shape[2], X_shape[3])
    y_shape = (T_length, y_shape[1], y_shape[2], y_shape[3])

    # handle save dir
    SAVE_FOLDER_DIR = SAVE_Base_DIR
    if not os.path.exists(SAVE_FOLDER_DIR):
        os.makedirs(SAVE_FOLDER_DIR)
    TARGET_X_DIR = os.path.join(SAVE_FOLDER_DIR, f'X_extra_layer_year_{year_list_to_str(year_range)}_{source_label}.zarr')
    TARGET_y_DIR = os.path.join(SAVE_FOLDER_DIR, f'{y_name}_year_{year_list_to_str(year_range)}_{source_label}.zarr')
    #print(X_shape)
    #print(y_shape)
    #print(TARGET_X_DIR)
    #print(TARGET_y_DIR)
    #print((1,X_shape[1], X_shape[2], X_shape[3]))
    

    # Use float64 so that the data is not compressed
    compressor = Blosc(cname='lz4',clevel=2,shuffle=1)
    X = None
    if IGNORE_X == False:
        X = zarr.open(
            TARGET_X_DIR,
            mode = 'w',
            shape=(X_shape[0],X_shape[1]+len(OBS_Name), X_shape[2], X_shape[3]),
            chunks=(1,X_shape[1]+len(OBS_Name), X_shape[2], X_shape[3]),
            dtype = np.float32,
            compressor=compressor
        )
    y = zarr.open(
        TARGET_y_DIR,
        mode = 'w',
        shape=y_shape,
        chunks=(1,y_shape[1], y_shape[2], y_shape[3]),
        dtype = np.float32,
        compressor=compressor
    )

    if IGNORE_X == False:
        print(f'Shape of X: {X.shape}')
    print(f'Shape of y: {y.shape}')

    prev_end = 0
    total_end = 0

    TIMESTAMP_PATH = os.path.join(SAVE_FOLDER_DIR, f'TIMESTAMP_{year_list_to_str(year_range)}_{source_label}.npy')
    timestamp_list = []
    for i in range(len(year_range)):
        c_year = year_range[i]
        tmp_X_DIR = os.path.join(input_base_DIR, f'X_{c_year}_{source_label}.zarr')
        tmp_y_DIR = os.path.join(input_base_DIR, f'{y_name}_{c_year}_{source_label}.zarr')
        tmp_OBS_DIR_list = []
        for name in OBS_Name:
            tmp_OBS_DIR_list.append(os.path.join(input_base_DIR, f'hmiHeader_{name}_{c_year}_{source_label}.zarr'))
 
        tmp_timestamp_DIR = os.path.join(input_base_DIR, f'TIMESTAMP_{c_year}_{source_label}.npy')
        tmp_timestamp_arr = np.load(tmp_timestamp_DIR)        
        timestamp_list.append(tmp_timestamp_arr)
        
        if IGNORE_X == False:
            print(tmp_X_DIR)
            tmp_X_zarr = zarr.open(tmp_X_DIR, mode='r')
        print(tmp_y_DIR)
        tmp_y_zarr = zarr.open(tmp_y_DIR, mode='r')
        if IGNORE_X == False:
            assert(tmp_X_zarr.shape[0] == tmp_y_zarr.shape[0])
        
        total_end += tmp_y_zarr.shape[0]
        print(f'{prev_end}, {total_end}\n')
        
        # write data for X,y
        if IGNORE_X == False:
            for i in range(len(tmp_OBS_DIR_list)):
                tmp_OBS_zarr = zarr.open(tmp_OBS_DIR_list[i], mode='r')
                
                X[prev_end:total_end, X_shape[1]+i, :, :] = np.broadcast_to((np.array(tmp_OBS_zarr))[:,np.newaxis, np.newaxis], (tmp_OBS_zarr.shape[0], GLOBAL_PADDING_SIZE, GLOBAL_PADDING_SIZE+1))
            X[prev_end:total_end, :X_shape[1], :, :] = tmp_X_zarr
            
        y[prev_end:total_end] = tmp_y_zarr
        prev_end = total_end
    
    
    timestamp_comb = np.concatenate(timestamp_list, axis=0)
    np.save(TIMESTAMP_PATH, timestamp_comb)
    assert(total_end == T_length)
    print(f'{total_end}, {T_length}')    




def rlh_preprocess_Data(
    srcBase: str,
    srcFlowBase: str,
    targetBase: str,
    c_year: str,
    DEBUG_MODE = False,
    QUIET_MODE = False,
    ZARR_VERIFY = True
    ):
    ########
    # Important Args:
    # c_year: The chosen (half) year for training dataset
    #         2011_01: 1-6
    #         2011_02: 7-12
    # 
    # Directory:
    #   target_dir: The base target directory for saved zarrs
    #   
    ########
    # the year chosen to be input
    target_dir = targetBase    
    
    
    verify_c_year(c_year)


    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    if QUIET_MODE == False:
        print(f'Loading data for year: {c_year}')
        print()

    # list of name of all folder that starts with c_year
    #Base_folder_list = [foldername for foldername in sorted(os.listdir(srcBase)) if
    #    foldername.startswith(c_year)]
    # !!! Folders in base and flowbase has the same name
    
    FlowBase_folder_list = None
    if int(c_year[-1]) == 1:
        FlowBase_folder_list = [foldername for foldername in sorted(os.listdir(srcFlowBase)) if
            foldername.startswith(c_year[:4]) and int(foldername[4:6])<=6]    
    if int(c_year[-1]) == 2:
        FlowBase_folder_list = [foldername for foldername in sorted(os.listdir(srcFlowBase)) if
            foldername.startswith(c_year[:4]) and int(foldername[4:6])>6]
    
    key = 'DISK'
    if 'Pole' in srcBase:
        key = 'Pole'

    # The length of all zarr
    Global_length_zarr = 0
    # Get the correct length of zarr (Exclude those with no npy)
    for i_main in range(len(FlowBase_folder_list)):
        foldername = FlowBase_folder_list[i_main]
        # use src to access file in folder for each measurement
        # src: srcBase and flowBase_folder list
        src = os.path.join(srcBase, foldername)
        
        
        for obsDate in sorted(os.listdir(src)):
            if obsDate.find("_") == -1 or not os.path.isdir(os.path.join(src,obsDate)):
                continue
            volFile = os.path.join(src, obsDate, obsDate+"_hmiObs.npz")
            volInvFile = os.path.join(src, obsDate, obsDate+"_hmiInv.npz")
            volSPInvFile = os.path.join(srcFlowBase, foldername, obsDate, obsDate+"_flowedSPInv.npz")
            volSPDisambigFile = os.path.join(srcFlowBase, foldername, obsDate, obsDate+"_flowedSPDisambig.npz")
            
            
            # check if file exists, if not, ignore this folder
            if os.path.exists(volFile) == False:
                ERR_MESSAGE_DIR = os.path.join(target_dir, f'ERROR_{c_year}_{key}.txt')
                with open(ERR_MESSAGE_DIR, 'a') as f:
                    f.write(volFile)
                    f.write('\n')
                continue
            
            if os.path.exists(volSPInvFile) == False:
                ERR_MESSAGE_DIR = os.path.join(target_dir, f'ERROR_{c_year}_{key}.txt')
                with open(ERR_MESSAGE_DIR, 'a') as f:
                    f.write(volSPInvFile)
                    f.write('\n')
                continue
            
            if os.path.exists(volSPDisambigFile) == False:
                ERR_MESSAGE_DIR = os.path.join(target_dir, f'ERROR_{c_year}_{key}.txt')
                with open(ERR_MESSAGE_DIR, 'a') as f:
                    f.write(volSPDisambigFile)
                    f.write('\n')
                continue
            
            Global_length_zarr += 1
    
    

    
    
    # If file already exists, ask if override
    # Will not override except enter 0
    if os.path.exists(os.path.join(target_dir, f'X_{c_year}_{key}.zarr')):
        print(f'File already exists, sure to OVERRIDE?')
        print(f' To continue, enter 0')
        tmp = input()
        if tmp != '0':
            print('No override, exit program now.')
            quit()
         
         
    # Variables to be saved 
    # Everything is saved as N,C,H,W 
    
    # Use float64 so that the data is not compressed
    compressor = Blosc(cname='lz4',clevel=2,shuffle=1)
    # Hardcode 24 channels for IQUV
    ZARR_comb_stoke = zarr.open(
        os.path.join(target_dir, f'X_{c_year}_{key}.zarr'),
        mode = 'w',
        shape=(Global_length_zarr, 25, GLOBAL_PADDING_SIZE, GLOBAL_PADDING_SIZE+1),
        chunks=(1,25, GLOBAL_PADDING_SIZE, GLOBAL_PADDING_SIZE+1),
        dtype = np.float32,
        compressor=compressor
    )
    
    ZARR_time_error = zarr.open(
        os.path.join(target_dir, f'timeError_{c_year}_{key}.zarr'),
        mode = 'w',
        shape=(Global_length_zarr, 1, GLOBAL_PADDING_SIZE, GLOBAL_PADDING_SIZE+1),
        chunks=(1,1, GLOBAL_PADDING_SIZE, GLOBAL_PADDING_SIZE+1),
        dtype = np.float32,
        compressor=compressor
    )
    # Not N,C,H,W, just N
    
    #hmi_heliographic_gt_label = {}
    
    initialized = False
    tmp_number = 0
    # First Initialize all the ZARRS in the dictionary
    
                
    print()      
    print('Initialization Zarr Complete')
    print()
    
    
    TIMESTAMP_list = []
    foldername = None
    
    ZARR_i = 0
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
            volInvFile = os.path.join(src, obsDate, obsDate+"_hmiInv.npz")
            
            
            
            # check if file exists, if not, ignore this folder
            if os.path.exists(volFile) == False:
                
                continue
            
            if os.path.exists(volSPInvFile) == False:
                
                continue

            if os.path.exists(volSPDisambigFile) == False:
                
                continue
            
            vol = np.load(volFile)
            volInv = np.load(volInvFile)
            
            
            

                
            tmp_key = None
            
            
            '''
            # Ignore for now
            # !!! hmi Inversion gt_labels
            for i in range(volInv['hmiInversionNames'].size):
                tmp_key = volInv['hmiInversionNames'][i]
                if not tmp_key in hmiInv_gt_label.keys():
                    hmiInv_gt_label[tmp_key] = []
                hmiInv_gt_label[tmp_key].append(pre_pad(volInv['hmiInversion'][:,:,i]))
            '''
            


            
            '''
            # Ignore for now
            # !!! hmi Heliographic Metadata
            for i in range(vol['hmiMetaHeliographic'].shape[2]):
                tmp_key = i
                if not tmp_key in hmi_heliographic_gt_label.keys():
                    hmi_heliographic_gt_label[tmp_key] = []
                hmi_heliographic_gt_label[tmp_key].append(pre_pad(vol['hmiMetaHeliographic'][:,:,i]))
            '''
            
            # !!! HMI Stokes, add continuum at the back!!!!!!!!!
            stoke = vol['hmiStokes']
            pad_stoke = np.transpose(pre_pad(stoke), (2,0,1))
            pad_continuum = np.transpose(pre_pad(volInv['con']), (2,0,1))
            ZARR_comb_stoke[ZARR_i, :24, :, :] = pad_stoke
            ZARR_comb_stoke[ZARR_i, 25, :, :] =  pad_stoke
            #print(np.array_equal(ZARR_comb_stoke[i_main, :, :, :], np.transpose(pre_pad(stoke), (2,0,1))))
            
            
            TIMESTAMP_list.append((obsDate, ZARR_i, src, srcFlowBase))
            
            
            ZARR_i += 1
            

    TIMESTAMP_list = np.array(TIMESTAMP_list)
    TIMESTAMP_PATH = os.path.join(target_dir, f'TIMESTAMP_{c_year}_{key}.npy')
    np.save(TIMESTAMP_PATH, TIMESTAMP_list)
    
            
            # Ignore for now
            # !!! HMI Continuum
            #comb_hmi_continuum.append(pre_pad(vol['hmiContinuum'][:,:,0]))
            
            