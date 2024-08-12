import os
import zarr
from numcodecs import Blosc
import numpy as np
from data_preprocessing import *

from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from datetime import datetime
from train import *
from pred_utils import *
from visualize_utils import *
from write_html_utils import *




import tmp_data_dir
from loguru import logger
import time
import os.path as osp
import os
import pdb
import getpass
import tempfile

###
# Folder Structures:
# !!! Pipeline and Extra layer is separated by MAIN_DIR
#   MAIN_DIR:
#       Training_Data:
#           year_[]:
#               Shared:
#                   X_year_[]
#               spInv_aB
#       Val_Data:
#           year_[]:
#               Shared:
#               spInv_aB               
###
from train_disk_only_version import *






GLOBAL_DATA_TYPE = ['pipeline', 'extra_layer']






# merge the disk and pole zarr, can choose whether to merge timestamp
def merge_Pole_Disk_zarr(
    DISK_ZARR_DIR:str, 
    POLE_ZARR_DIR:str,
    TARGET_PATH:str,
    DISK_TIMESTAMP_NPY_DIR:str = None,
    POLE_TIMESTAMP_NPY_DIR:str = None, 
    ):
    
    assert('DISK' in DISK_ZARR_DIR)
    assert('Pole' in POLE_ZARR_DIR)
    Disk_zarr = zarr.open(DISK_ZARR_DIR, mode='r')
    Pole_zarr = zarr.open(POLE_ZARR_DIR, mode='r')
    T_length = Disk_zarr.shape[0] + Pole_zarr.shape[0]
    Disk_shape = Disk_zarr.shape
    Pole_shape = Pole_zarr.shape
    R_shape = (T_length,)
    if len(Disk_shape) != 1:
        for i in range(1,4):
            assert(Disk_shape[i] == Pole_shape[i])
        R_shape = (T_length, Disk_shape[1], Disk_shape[2], Disk_shape[3])
    if not os.path.exists(TARGET_PATH):
        os.makedirs(TARGET_PATH)
    
    compressor = Blosc(cname='lz4',clevel=2,shuffle=1)

    filename = DISK_ZARR_DIR.split("/")
    filename = (filename[-1]).replace("DISK", "COMB")
    if len(Disk_shape) == 1:
        chunk_shape = (1,)
    else:
        chunk_shape = (1, Disk_shape[1], Disk_shape[2], Disk_shape[3])

    assert(filename == ((POLE_ZARR_DIR.split("/"))[-1]).replace("Pole", "COMB"))
    Comb_zarr = zarr.open(
        os.path.join(TARGET_PATH, filename),
        mode = 'w',
        shape = R_shape,
        chunks = chunk_shape,
        dtype = np.float64,
        compressor = compressor,
    )
    
    # Always Disk first, Pole second
    
    prev_end = Disk_zarr.shape[0]
    Comb_zarr[:prev_end] = Disk_zarr
    Comb_zarr[prev_end:] = Pole_zarr
    
    # Process the timestamp array
    if DISK_TIMESTAMP_NPY_DIR != None:
        DISK_NPY = np.load(DISK_TIMESTAMP_NPY_DIR)
        POLE_NPY = np.load(POLE_TIMESTAMP_NPY_DIR)
        TIMESTAMP_COMB = np.concatenate((DISK_NPY, POLE_NPY), axis=0)

        timestamp_name = POLE_TIMESTAMP_NPY_DIR.split("/")[-1]
        
        
        timestamp_name = timestamp_name.replace("Pole", "COMB")
        np.save(os.path.join(TARGET_PATH, timestamp_name), TIMESTAMP_COMB)
    


class Name_Manage():
    def __init__(
        self, 
        DISK_POOL:str,
        POLE_POOL:str,
        MAIN_DIR:str, 
        NFS_MODEL_SAVE_DIR:str,
        PREDICT_SAVE_DIR:str,
        train_year_list,
        val_year_list,
        test_year_list,
        ):
        #self.srcBase_DISK = srcBase_DISK
        #self.srcFlowBase_DISK = srcFlowBase_DISK
        #self.srcBase_POLE = srcBase_POLE
        #self.srcFlowBase_POLE = srcFlowBase_POLE
        self.DISK_POOL = DISK_POOL
        self.POLE_POOL = POLE_POOL
        self.MAIN_DIR = MAIN_DIR
        if not os.path.exists(self.MAIN_DIR):
            os.makedirs(self.MAIN_DIR)
        self.train_year_list = train_year_list
        self.val_year_list = val_year_list
        self.test_year_list = test_year_list
        self.NFS_MODEL_SAVE_DIR = NFS_MODEL_SAVE_DIR
        
        self.PREDICT_SAVE_DIR = PREDICT_SAVE_DIR
        self.Train_data_DIR = os.path.join(MAIN_DIR, 'TRAIN_DATA')
        self.Train_data_DIR = os.path.join(self.Train_data_DIR, f'year_{year_list_to_str(train_year_list)}')
        self.VAL_data_DIR = os.path.join(MAIN_DIR, 'VAL_DATA')
        self.VAL_data_DIR = os.path.join(self.VAL_data_DIR, f'year_{year_list_to_str(val_year_list)}')
        self.Test_data_DIR = os.path.join(MAIN_DIR, 'TEST_DATA')
        self.Test_data_DIR = os.path.join(self.Test_data_DIR, f'year_{year_list_to_str(test_year_list)}')
        if not os.path.exists(self.Train_data_DIR):
            os.makedirs(self.Train_data_DIR)
        if not os.path.exists(self.VAL_data_DIR):
            os.makedirs(self.VAL_data_DIR)
        if not os.path.exists(self.Test_data_DIR):
            os.makedirs(self.Test_data_DIR)

    def build_train_dataset(self, y:str, data_type:str):
        assert(data_type in GLOBAL_DATA_TYPE)
        Train_X_DIR = os.path.join(self.Train_data_DIR, 'Shared')
        Val_X_DIR = os.path.join(self.VAL_data_DIR, 'Shared')
        Test_X_DIR = os.path.join(self.Test_data_DIR, 'Shared')
        Train_y_DIR = os.path.join(self.Train_data_DIR, f'{y}')
        Val_y_DIR = os.path.join(self.VAL_data_DIR, f'{y}')
        Test_y_DIR = os.path.join(self.Test_data_DIR, f'{y}')
        #TMP_ZARR_DIR = os.path.join()
        utils_make_dir(Train_X_DIR)
        utils_make_dir(Val_X_DIR)
        utils_make_dir(Test_X_DIR)
        utils_make_dir(Train_y_DIR)
        utils_make_dir(Val_y_DIR)
        utils_make_dir(Test_y_DIR)

        

        def tmp_build_dataset(
            key:str,
            year_list:list,
            X_DIR:str,
            y_DIR:str,
            data_type:str,
            ):
            assert(key in ['TRAIN', 'VAL', 'TEST'])
            assert(data_type in GLOBAL_DATA_TYPE)
            X_ZARR_DIR = os.path.join(X_DIR, f'X_{data_type}_year_{year_list_to_str(year_list)}_COMB.zarr')
            print(X_ZARR_DIR)
            y_ZARR_DIR = os.path.join(y_DIR, f'{y}_year_{year_list_to_str(year_list)}_COMB.zarr')
            print(y_ZARR_DIR)

            
            # If not, build the DISK, Pole dataset separately into tmp folder
            if not os.path.exists(X_ZARR_DIR):
                # TMP folder for two part
                TMP_TWO_PART_PATH = os.path.join(self.MAIN_DIR, f'TMP_X_Store')
                
                # Disk
                preprocess_generate_pipeline(
                    input_base_DIR= self.DISK_POOL,
                    year_range=year_list,
                    y_name='hmiMetadata_2',
                    SAVE_Base_DIR=TMP_TWO_PART_PATH,
                    VALIDATION=True,
                    IGNORE_X=False,
                )

                
                # Pole
                preprocess_generate_pipeline(
                    input_base_DIR= self.POLE_POOL,
                    year_range=year_list,
                    y_name='hmiMetadata_2',
                    SAVE_Base_DIR=TMP_TWO_PART_PATH,
                    VALIDATION=True,
                    IGNORE_X=False,
                )
                

                # Combine these
                # Combine X
                TMP_Disk_X_ZARR_DIR = os.path.join(TMP_TWO_PART_PATH, f'X_{data_type}_year_{year_list_to_str(year_list)}_DISK.zarr')
                TMP_Pole_X_ZARR_DIR = os.path.join(TMP_TWO_PART_PATH, f'X_{data_type}_year_{year_list_to_str(year_list)}_Pole.zarr')
                TMP_Disk_timestamp_DIR = os.path.join(TMP_TWO_PART_PATH, f'TIMESTAMP_{year_list_to_str(year_list)}_DISK.npy')
                TMP_Pole_timestamp_DIR = os.path.join(TMP_TWO_PART_PATH, f'TIMESTAMP_{year_list_to_str(year_list)}_Pole.npy')
                merge_Pole_Disk_zarr(TMP_Disk_X_ZARR_DIR, TMP_Pole_X_ZARR_DIR, X_DIR, TMP_Disk_timestamp_DIR, TMP_Pole_timestamp_DIR)

                # Combine hmiMetadata_2
                TMP_Disk_hmiMetadata_ZARR_DIR = os.path.join(TMP_TWO_PART_PATH, f'hmiMetadata_2_year_{year_list_to_str(year_list)}_DISK.zarr')
                TMP_Pole_hmiMetadata_ZARR_DIR = os.path.join(TMP_TWO_PART_PATH, f'hmiMetadata_2_year_{year_list_to_str(year_list)}_Pole.zarr')
                merge_Pole_Disk_zarr(TMP_Disk_hmiMetadata_ZARR_DIR, TMP_Pole_hmiMetadata_ZARR_DIR, X_DIR)

                
                # For OBS_VN
                # Disk
                preprocess_generate_pipeline(
                    input_base_DIR= self.DISK_POOL,
                    year_range=year_list,
                    y_name='hmiHeader_OBS_VN',
                    SAVE_Base_DIR=TMP_TWO_PART_PATH,
                    VALIDATION=True,
                    IGNORE_X=True,
                )
                # Pole
                preprocess_generate_pipeline(
                    input_base_DIR= self.POLE_POOL,
                    year_range=year_list,
                    y_name='hmiHeader_OBS_VN',
                    SAVE_Base_DIR=TMP_TWO_PART_PATH,
                    VALIDATION=True,
                    IGNORE_X=True,
                )
                TMP_Disk_hmiMetadata_ZARR_DIR = os.path.join(TMP_TWO_PART_PATH, f'hmiHeader_OBS_VN_year_{year_list_to_str(year_list)}_DISK.zarr')
                TMP_Pole_hmiMetadata_ZARR_DIR = os.path.join(TMP_TWO_PART_PATH, f'hmiHeader_OBS_VN_year_{year_list_to_str(year_list)}_Pole.zarr')
                merge_Pole_Disk_zarr(TMP_Disk_hmiMetadata_ZARR_DIR, TMP_Pole_hmiMetadata_ZARR_DIR, X_DIR)

                # For OBS_VR
                # Disk
                preprocess_generate_pipeline(
                    input_base_DIR= self.DISK_POOL,
                    year_range=year_list,
                    y_name='hmiHeader_OBS_VR',
                    SAVE_Base_DIR=TMP_TWO_PART_PATH,
                    VALIDATION=True,
                    IGNORE_X=True,
                )
                # Pole
                preprocess_generate_pipeline(
                    input_base_DIR= self.POLE_POOL,
                    year_range=year_list,
                    y_name='hmiHeader_OBS_VR',
                    SAVE_Base_DIR=TMP_TWO_PART_PATH,
                    VALIDATION=True,
                    IGNORE_X=True,
                )
                TMP_Disk_hmiMetadata_ZARR_DIR = os.path.join(TMP_TWO_PART_PATH, f'hmiHeader_OBS_VR_year_{year_list_to_str(year_list)}_DISK.zarr')
                TMP_Pole_hmiMetadata_ZARR_DIR = os.path.join(TMP_TWO_PART_PATH, f'hmiHeader_OBS_VR_year_{year_list_to_str(year_list)}_Pole.zarr')
                merge_Pole_Disk_zarr(TMP_Disk_hmiMetadata_ZARR_DIR, TMP_Pole_hmiMetadata_ZARR_DIR, X_DIR)


                # For OBS_VW
                # Disk
                preprocess_generate_pipeline(
                    input_base_DIR= self.DISK_POOL,
                    year_range=year_list,
                    y_name='hmiHeader_OBS_VW',
                    SAVE_Base_DIR=TMP_TWO_PART_PATH,
                    VALIDATION=True,
                    IGNORE_X=True,
                )
                # Pole
                preprocess_generate_pipeline(
                    input_base_DIR= self.POLE_POOL,
                    year_range=year_list,
                    y_name='hmiHeader_OBS_VW',
                    SAVE_Base_DIR=TMP_TWO_PART_PATH,
                    VALIDATION=True,
                    IGNORE_X=True,
                )
                TMP_Disk_hmiMetadata_ZARR_DIR = os.path.join(TMP_TWO_PART_PATH, f'hmiHeader_OBS_VW_year_{year_list_to_str(year_list)}_DISK.zarr')
                TMP_Pole_hmiMetadata_ZARR_DIR = os.path.join(TMP_TWO_PART_PATH, f'hmiHeader_OBS_VW_year_{year_list_to_str(year_list)}_Pole.zarr')
                merge_Pole_Disk_zarr(TMP_Disk_hmiMetadata_ZARR_DIR, TMP_Pole_hmiMetadata_ZARR_DIR, X_DIR)

                

            y_ZARR_DIR = os.path.join(y_DIR, f'{y}_year_{year_list_to_str(year_list)}_COMB.zarr')
            if not os.path.exists(y_ZARR_DIR):
                TMP_TWO_PART_PATH = os.path.join(self.MAIN_DIR, f'TMP_{y}_Store')
                # Disk
                preprocess_generate_pipeline(
                    input_base_DIR= self.DISK_POOL,
                    year_range=year_list,
                    y_name=y,
                    SAVE_Base_DIR=TMP_TWO_PART_PATH,
                    VALIDATION=True,
                    IGNORE_X=True,
                )
                
                # Pole
                preprocess_generate_pipeline(
                    input_base_DIR= self.POLE_POOL,
                    year_range=year_list,
                    y_name=y,
                    SAVE_Base_DIR=TMP_TWO_PART_PATH,
                    VALIDATION=True,
                    IGNORE_X=True,
                )
                TMP_Disk_y_ZARR_DIR = os.path.join(TMP_TWO_PART_PATH, f'{y}_year_{year_list_to_str(year_list)}_DISK.zarr')
                TMP_Pole_y_ZARR_DIR = os.path.join(TMP_TWO_PART_PATH, f'{y}_year_{year_list_to_str(year_list)}_Pole.zarr')
                TMP_Disk_timestamp_DIR = os.path.join(TMP_TWO_PART_PATH, f'TIMESTAMP_{year_list_to_str(year_list)}_DISK.npy')
                TMP_Pole_timestamp_DIR = os.path.join(TMP_TWO_PART_PATH, f'TIMESTAMP_{year_list_to_str(year_list)}_Pole.npy')
                merge_Pole_Disk_zarr(TMP_Disk_y_ZARR_DIR, TMP_Pole_y_ZARR_DIR, y_DIR, TMP_Disk_timestamp_DIR, TMP_Pole_timestamp_DIR)
            
            return X_ZARR_DIR, y_ZARR_DIR
        
        tmp_build_dataset('TRAIN', self.train_year_list, Train_X_DIR, Train_y_DIR, data_type)
        
        print('Train complete')
        #x = input()
        
        tmp_build_dataset('VAL', self.val_year_list, Val_X_DIR, Val_y_DIR, data_type)
        
        print('Val complete')
        #x = input()
        
        tmp_build_dataset('TEST', self.test_year_list, Test_X_DIR, Test_y_DIR, data_type)
        




    # Mind Disk Mask
    def train(
        self, 
        y:str, 
        data_type:str, 
        GLOBAL_DEVICE, 
        lr, 
        weight_decay, 
        eps, 
        model, 
        TAR_FILE_DIR:str = None,
        OBS_list = None, # [hmiHeader_OBS_VN, hmiHeader_OBS_VR, hmiHeader_OBS_VW]
        bins = [],
        whether_continue_training = None,
        main_h_flip = False,
        ):
        # For transfer to SSD
        if TAR_FILE_DIR != None:
            tmpdir =  tmp_data_dir.TempSSDPath(TAR_FILE_DIR, 'test_dataset', logging=True)

        Train_X_DIR = os.path.join(self.Train_data_DIR, 'Shared')
        Val_X_DIR = os.path.join(self.VAL_data_DIR, 'Shared')
        #Test_X_DIR = os.path.join(self.Test_data_DIR, 'Shared')
        Train_y_DIR = os.path.join(self.Train_data_DIR, f'{y}')
        Val_y_DIR = os.path.join(self.VAL_data_DIR, f'{y}')
        #Test_y_DIR = os.path.join(self.Test_data_DIR, f'{y}')
        
        assert(data_type == 'pipeline')
        Train_X_ZARR_DIR = None
        Train_y_ZARR_DIR = None
        Val_X_ZARR_DIR = None
        Val_y_ZARR_DIR = None
        
        OBS_TRAIN_DIR_LIST = None
        OBS_TRAIN_NORM_DIR_LIST = None
        OBS_VAL_DIR_LIST = None
        if OBS_list != None:
            OBS_TRAIN_DIR_LIST = []
            OBS_TRAIN_NORM_DIR_LIST = []
            OBS_VAL_DIR_LIST = []
            for OBS_NAME in OBS_list:
                OBS_TRAIN_DIR_LIST.append(os.path.join(Train_X_DIR, f'{OBS_NAME}_year_{year_list_to_str(self.train_year_list)}_COMB.zarr'))
                OBS_TRAIN_NORM_DIR_LIST.append(os.path.join(Train_X_DIR, f'{OBS_NAME}_Norm_param.npy'))
                OBS_VAL_DIR_LIST.append(os.path.join(Val_X_DIR, f'{OBS_NAME}_year_{year_list_to_str(self.val_year_list)}_COMB.zarr'))
        

        Train_X_ZARR_DIR = os.path.join(Train_X_DIR, f'X_{data_type}_year_{year_list_to_str(self.train_year_list)}_COMB.zarr')
        Train_y_ZARR_DIR = os.path.join(Train_y_DIR, f'{y}_year_{year_list_to_str(self.train_year_list)}_COMB.zarr')
        Val_X_ZARR_DIR = os.path.join(Val_X_DIR, f'X_{data_type}_year_{year_list_to_str(self.val_year_list)}_COMB.zarr')
        Val_y_ZARR_DIR = os.path.join(Val_y_DIR, f'{y}_year_{year_list_to_str(self.val_year_list)}_COMB.zarr')
        
        Train_OBS_VN_ZARR_DIR = os.path.join(Train_X_DIR, f'hmiHeader_OBS_VN_year_{year_list_to_str(self.train_year_list)}_COMB.zarr')
        Val_OBS_VN_ZARR_DIR = os.path.join(Val_X_DIR, f'hmiHeader_OBS_VN_year_{year_list_to_str(self.val_year_list)}_COMB.zarr')
        Train_OBS_VR_ZARR_DIR = os.path.join(Train_X_DIR, f'hmiHeader_OBS_VR_year_{year_list_to_str(self.train_year_list)}_COMB.zarr')
        Val_OBS_VR_ZARR_DIR = os.path.join(Val_X_DIR, f'hmiHeader_OBS_VR_year_{year_list_to_str(self.val_year_list)}_COMB.zarr')
        Train_OBS_VW_ZARR_DIR = os.path.join(Train_X_DIR, f'hmiHeader_OBS_VW_year_{year_list_to_str(self.train_year_list)}_COMB.zarr')
        Val_OBS_VW_ZARR_DIR = os.path.join(Val_X_DIR, f'hmiHeader_OBS_VW_year_{year_list_to_str(self.val_year_list)}_COMB.zarr')
        
        X_NORM_DIR = os.path.join(Train_X_DIR, 'X_Norm_param.npy')
        y_NORM_DIR = os.path.join(Train_y_DIR, 'y_Norm_param.npy')
        
        #import time
        #time.sleep(100000000)

        # Save normalization parameters in Train_y_DIR
        #print('Calculating parameters')
        
        if not os.path.exists(X_NORM_DIR):
            print(f'Calculating parameters for X')
            save_normalization_parameter_OBS_V(
                Train_DIR = Train_OBS_VN_ZARR_DIR, 
                Val_DIR = Val_OBS_VN_ZARR_DIR,
                SAVE_DIR = Train_X_DIR,
                name='hmiHeader_OBS_VN',
                SAMPLE_SIZE = 400,
                )
            save_normalization_parameter_OBS_V(
                Train_DIR = Train_OBS_VR_ZARR_DIR, 
                Val_DIR = Val_OBS_VR_ZARR_DIR,
                SAVE_DIR = Train_X_DIR,
                name='hmiHeader_OBS_VR',
                SAMPLE_SIZE = 400,
                )
            save_normalization_parameter_OBS_V(
                Train_DIR = Train_OBS_VW_ZARR_DIR, 
                Val_DIR = Val_OBS_VW_ZARR_DIR,
                SAVE_DIR = Train_X_DIR,
                name='hmiHeader_OBS_VW',
                SAMPLE_SIZE = 400,
                )
            
            
            save_normalization_parameter(
                Train_DIR = Train_X_ZARR_DIR, 
                Val_DIR = Val_X_ZARR_DIR,
                SAVE_DIR = Train_X_DIR,
                name='X',
                SAMPLE_SIZE = 400,
                )
            
            
        if not os.path.exists(y_NORM_DIR):
            save_normalization_parameter(
                Train_DIR = Train_y_ZARR_DIR, 
                Val_DIR = Val_y_ZARR_DIR,
                SAVE_DIR = Train_y_DIR,
                name='y',
                SAMPLE_SIZE = 400,
                )
        
        print('Saved Normalization')

        
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, eps=eps)
        rlrop = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
        
        DISK_MASK_TRAIN_DIR = os.path.join(Train_X_DIR, f'hmiMetadata_2_year_{year_list_to_str(self.train_year_list)}_COMB.zarr')
        DISK_MASK_VAL_DIR = os.path.join(Val_X_DIR, f'hmiMetadata_2_year_{year_list_to_str(self.val_year_list)}_COMB.zarr')
        
        if OBS_list == None:
            model_type = 'pipeline'
        else:
            model_type = year_list_to_str(OBS_list)

        MODEL_SAVE_PATH = os.path.join(self.NFS_MODEL_SAVE_DIR, f'{model_type}_MODEL')
        MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, y)
        
        train(
            Train_X_ZARR_DIR,
            Val_X_ZARR_DIR,
            Train_y_ZARR_DIR,
            Val_y_ZARR_DIR,
            X_NORM_DIR,
            y_NORM_DIR,
            DISK_MASK_TRAIN_DIR,
            DISK_MASK_VAL_DIR,
            model,
            optimizer,
            rlrop,
            MODEL_SAVE_PATH,
            GLOBAL_DEVICE,
            lr,
            OBS_TRAIN_DIR_LIST,
            OBS_TRAIN_NORM_DIR_LIST,
            OBS_VAL_DIR_LIST,
            bins,
            whether_continue_training,
            main_h_flip,
            )
        #tmpdir.clean_up()
    
    
    
    # Mind Disk Mask
    def train_disk_only(
        self, 
        y:str, 
        data_type:str, 
        GLOBAL_DEVICE, 
        lr, 
        weight_decay, 
        eps, 
        model, 
        TAR_FILE_DIR:str = None,
        OBS_list = None, # [hmiHeader_OBS_VN, hmiHeader_OBS_VR, hmiHeader_OBS_VW]
        bins = [],
        whether_continue_training = None,
        ):
        # For transfer to SSD
        if TAR_FILE_DIR != None:
            tmpdir =  tmp_data_dir.TempSSDPath(TAR_FILE_DIR, 'test_dataset', logging=True)

        Train_X_DIR = os.path.join(self.Train_data_DIR, 'Shared')
        Val_X_DIR = os.path.join(self.VAL_data_DIR, 'Shared')
        #Test_X_DIR = os.path.join(self.Test_data_DIR, 'Shared')
        Train_y_DIR = os.path.join(self.Train_data_DIR, f'{y}')
        Val_y_DIR = os.path.join(self.VAL_data_DIR, f'{y}')
        #Test_y_DIR = os.path.join(self.Test_data_DIR, f'{y}')
        
        assert(data_type == 'pipeline')
        Train_X_ZARR_DIR = None
        Train_y_ZARR_DIR = None
        Val_X_ZARR_DIR = None
        Val_y_ZARR_DIR = None
        
        OBS_TRAIN_DIR_LIST = None
        OBS_TRAIN_NORM_DIR_LIST = None
        OBS_VAL_DIR_LIST = None
        if OBS_list != None:
            OBS_TRAIN_DIR_LIST = []
            OBS_TRAIN_NORM_DIR_LIST = []
            OBS_VAL_DIR_LIST = []
            for OBS_NAME in OBS_list:
                OBS_TRAIN_DIR_LIST.append(os.path.join(Train_X_DIR, f'{OBS_NAME}_year_{year_list_to_str(self.train_year_list)}_COMB.zarr'))
                OBS_TRAIN_NORM_DIR_LIST.append(os.path.join(Train_X_DIR, f'{OBS_NAME}_Norm_param.npy'))
                OBS_VAL_DIR_LIST.append(os.path.join(Val_X_DIR, f'{OBS_NAME}_year_{year_list_to_str(self.val_year_list)}_COMB.zarr'))
        

        Train_X_ZARR_DIR = os.path.join(Train_X_DIR, f'X_{data_type}_year_{year_list_to_str(self.train_year_list)}_COMB.zarr')
        Train_y_ZARR_DIR = os.path.join(Train_y_DIR, f'{y}_year_{year_list_to_str(self.train_year_list)}_COMB.zarr')
        Val_X_ZARR_DIR = os.path.join(Val_X_DIR, f'X_{data_type}_year_{year_list_to_str(self.val_year_list)}_COMB.zarr')
        Val_y_ZARR_DIR = os.path.join(Val_y_DIR, f'{y}_year_{year_list_to_str(self.val_year_list)}_COMB.zarr')
        
        Train_OBS_VN_ZARR_DIR = os.path.join(Train_X_DIR, f'hmiHeader_OBS_VN_year_{year_list_to_str(self.train_year_list)}_COMB.zarr')
        Val_OBS_VN_ZARR_DIR = os.path.join(Val_X_DIR, f'hmiHeader_OBS_VN_year_{year_list_to_str(self.val_year_list)}_COMB.zarr')
        Train_OBS_VR_ZARR_DIR = os.path.join(Train_X_DIR, f'hmiHeader_OBS_VR_year_{year_list_to_str(self.train_year_list)}_COMB.zarr')
        Val_OBS_VR_ZARR_DIR = os.path.join(Val_X_DIR, f'hmiHeader_OBS_VR_year_{year_list_to_str(self.val_year_list)}_COMB.zarr')
        Train_OBS_VW_ZARR_DIR = os.path.join(Train_X_DIR, f'hmiHeader_OBS_VW_year_{year_list_to_str(self.train_year_list)}_COMB.zarr')
        Val_OBS_VW_ZARR_DIR = os.path.join(Val_X_DIR, f'hmiHeader_OBS_VW_year_{year_list_to_str(self.val_year_list)}_COMB.zarr')
        
        X_NORM_DIR = os.path.join(Train_X_DIR, 'X_Norm_param.npy')
        y_NORM_DIR = os.path.join(Train_y_DIR, 'y_Norm_param.npy')

        # Save normalization parameters in Train_y_DIR
        print('Calculating parameters')
        
        if not os.path.exists(X_NORM_DIR):
            print(f'Calculating parameters for X')
            save_normalization_parameter_OBS_V(
                Train_DIR = Train_OBS_VN_ZARR_DIR, 
                Val_DIR = Val_OBS_VN_ZARR_DIR,
                SAVE_DIR = Train_X_DIR,
                name='hmiHeader_OBS_VN',
                SAMPLE_SIZE = 400,
                )
            save_normalization_parameter_OBS_V(
                Train_DIR = Train_OBS_VR_ZARR_DIR, 
                Val_DIR = Val_OBS_VR_ZARR_DIR,
                SAVE_DIR = Train_X_DIR,
                name='hmiHeader_OBS_VR',
                SAMPLE_SIZE = 400,
                )
            save_normalization_parameter_OBS_V(
                Train_DIR = Train_OBS_VW_ZARR_DIR, 
                Val_DIR = Val_OBS_VW_ZARR_DIR,
                SAVE_DIR = Train_X_DIR,
                name='hmiHeader_OBS_VW',
                SAMPLE_SIZE = 400,
                )
            
            
            save_normalization_parameter(
                Train_DIR = Train_X_ZARR_DIR, 
                Val_DIR = Val_X_ZARR_DIR,
                SAVE_DIR = Train_X_DIR,
                name='X',
                SAMPLE_SIZE = 400,
                )
            
            
        if not os.path.exists(y_NORM_DIR):
            save_normalization_parameter(
                Train_DIR = Train_y_ZARR_DIR, 
                Val_DIR = Val_y_ZARR_DIR,
                SAVE_DIR = Train_y_DIR,
                name='y',
                SAMPLE_SIZE = 400,
                )
        
        

        
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, eps=eps)
        rlrop = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
        
        DISK_MASK_TRAIN_DIR = os.path.join(Train_X_DIR, f'hmiMetadata_2_year_{year_list_to_str(self.train_year_list)}_COMB.zarr')
        DISK_MASK_VAL_DIR = os.path.join(Val_X_DIR, f'hmiMetadata_2_year_{year_list_to_str(self.val_year_list)}_COMB.zarr')
        
        if OBS_list == None:
            model_type = 'pipeline'
        else:
            model_type = year_list_to_str(OBS_list)

        MODEL_SAVE_PATH = os.path.join(self.NFS_MODEL_SAVE_DIR, f'{model_type}_MODEL')
        MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, y)
        
        
        train_disk_only(
            Train_X_ZARR_DIR,
            Val_X_ZARR_DIR,
            Train_y_ZARR_DIR,
            Val_y_ZARR_DIR,
            X_NORM_DIR,
            y_NORM_DIR,
            DISK_MASK_TRAIN_DIR,
            DISK_MASK_VAL_DIR,
            model,
            optimizer,
            rlrop,
            MODEL_SAVE_PATH,
            GLOBAL_DEVICE,
            lr,
            OBS_TRAIN_DIR_LIST,
            OBS_TRAIN_NORM_DIR_LIST,
            OBS_VAL_DIR_LIST,
            bins,
            whether_continue_training,
            )
        #tmpdir.clean_up()
        



    def predict(self, y:str, data_type:str, GLOBAL_DEVICE, model, OBS_list = None, bins=[], whether_CI = False):
        assert(data_type == 'pipeline')
        if OBS_list == None:
            model_type = 'pipeline'
        else:
            model_type = year_list_to_str(OBS_list)

        MODEL_SAVE_PATH = os.path.join(self.NFS_MODEL_SAVE_DIR, f'{model_type}_MODEL')
        MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, y)
        
        arr = sorted(os.listdir(MODEL_SAVE_PATH))
        print()
        for i in range(len(arr)):
            print(f'{i}: {arr[i]}')
        print('Enter the index of model to load from:')
        index = int(input())
        MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, arr[index])
        print(f'Loading from {MODEL_SAVE_PATH}')
        model_name = arr[index]
        
        tmp_npy_dir = os.path.join(MODEL_SAVE_PATH, 'Train_loss.npy')
        loss_arr = np.load(tmp_npy_dir)
        print(f'The model has been trained for {len(loss_arr)} epoch')
        
        arr_list = [filename for filename in (os.listdir(MODEL_SAVE_PATH)) if filename.startswith('epoch')]
        
        for name in arr_list:
            print(name)
        
        print('Enter epoch number')
        epoch = int(input())

        model_out_path = os.path.join(MODEL_SAVE_PATH, f'epoch={epoch}.checkpoint.pth')
        print(f'Loading model for epoch {epoch} from {model_out_path}')

        model.load_state_dict(torch.load(model_out_path, map_location=GLOBAL_DEVICE)['model'])
        model.eval()

        Test_X_DIR = os.path.join(self.Test_data_DIR, 'Shared')
        Test_y_DIR = os.path.join(self.Test_data_DIR, f'{y}')

        Test_X_ZARR_DIR = os.path.join(Test_X_DIR, f'X_{data_type}_year_{year_list_to_str(self.test_year_list)}_COMB.zarr')
        Test_y_ZARR_DIR = os.path.join(Test_y_DIR, f'{y}_year_{year_list_to_str(self.test_year_list)}_COMB.zarr')

        DISK_MASK_DIR = os.path.join(Test_X_DIR, f'hmiMetadata_2_year_{year_list_to_str(self.test_year_list)}_COMB.zarr')
        
        Train_y_DIR = os.path.join(self.Train_data_DIR, f'{y}')
        Train_X_DIR = os.path.join(self.Train_data_DIR, 'Shared')
        X_NORM_DIR = os.path.join(Train_X_DIR, 'X_Norm_param.npy')
        y_NORM_DIR = os.path.join(Train_y_DIR, 'y_Norm_param.npy')
        
        OBS_TEST_DIR_LIST = None
        OBS_TRAIN_NORM_DIR_LIST = None
        
        if OBS_list != None:
            OBS_TEST_DIR_LIST = []
            OBS_TRAIN_NORM_DIR_LIST = []
            for OBS_NAME in OBS_list:
                OBS_TEST_DIR_LIST.append(os.path.join(Test_X_DIR, f'{OBS_NAME}_year_{year_list_to_str(self.test_year_list)}_COMB.zarr'))
                OBS_TRAIN_NORM_DIR_LIST.append(os.path.join(Train_X_DIR, f'{OBS_NAME}_Norm_param.npy'))

        test_dataset = HMI_Pipeline(Test_X_ZARR_DIR, None, X_NORM_DIR, None, DISK_MASK_DIR, 'test', GLOBAL_DEVICE, True, OBS_TEST_DIR_LIST, OBS_TRAIN_NORM_DIR_LIST)
        test_loader = DataLoader(test_dataset, batch_size=1, num_workers=8, pin_memory=False)

        PRED_DIR = self.PREDICT_SAVE_DIR
        PRED_DIR = os.path.join(PRED_DIR, f'{y}')
        PRED_DIR = os.path.join(PRED_DIR, model_type)
        PRED_DIR = os.path.join(PRED_DIR, model_name)
        PRED_DIR = os.path.join(PRED_DIR, f'epoch={epoch}')
        
        TIMESTAMP_DIR = os.path.join(Test_X_DIR, f'TIMESTAMP_{year_list_to_str(self.test_year_list)}_COMB.npy')
        if bins == []:
            print('Regression')
            pred_pipeline(test_loader, model, GLOBAL_DEVICE, PRED_DIR, y_NORM_DIR, TIMESTAMP_DIR)
        else:
            classification_predict_with_noise(test_loader, model, GLOBAL_DEVICE, PRED_DIR, TIMESTAMP_DIR, bins, whether_CI)

        
    def visualize(
        self, 
        y:str, 
        t, 
        img_idx_list, 
        OBS_list = None, 
        Pole_only_index = None,
        bin_class = None
        ):
        data_type = 'pipeline'
        if OBS_list == None:
            model_type = 'pipeline'
        else:
            model_type = year_list_to_str(OBS_list)
        
        NPY_SAVE_DIR = self.PREDICT_SAVE_DIR
        NPY_SAVE_DIR = os.path.join(NPY_SAVE_DIR, f'{y}')
        NPY_SAVE_DIR = os.path.join(NPY_SAVE_DIR, model_type)
        
        # Get the npy save DIR
        if True:
            arr = sorted(os.listdir(NPY_SAVE_DIR))
            print()
            for i in range(len(arr)):
                print(f'{i}: {arr[i]}')
            print('Enter the index of model to load from:')
            #index = int(input())
            index = 0
            model_name = arr[index]
        NPY_SAVE_DIR = os.path.join(NPY_SAVE_DIR, model_name)

        if True:
            arr = sorted(os.listdir(NPY_SAVE_DIR))
            print()
            for i in range(len(arr)):
                print(f'{i}: {arr[i]}')
            print('Enter the index of epoch to load from:')
            #index = int(input())
            index = 0
            epoch = arr[index]
        else:
            epoch = f'epoch={epoch}'
        NPY_SAVE_DIR = os.path.join(NPY_SAVE_DIR, epoch)

        VISUALIZE_PATH = (self.PREDICT_SAVE_DIR).replace("PREDICT", "VISUALIZE")
        
        VISUALIZE_PATH = os.path.join(VISUALIZE_PATH, f'{y}')
        VISUALIZE_PATH = os.path.join(VISUALIZE_PATH, f'{model_name}')
        VISUALIZE_PATH = os.path.join(VISUALIZE_PATH, f'epoch={epoch}')
        
        description = 'standard'

        if bin_class != None:
            description = bin_class._get_description()
        
        VISUALIZE_PATH = os.path.join(VISUALIZE_PATH, description)
        
        Test_X_DIR = os.path.join(self.Test_data_DIR, 'Shared')
        Test_y_DIR = os.path.join(self.Test_data_DIR, f'{y}')
        Test_y_ZARR_DIR = os.path.join(Test_y_DIR, f'{y}_year_{year_list_to_str(self.test_year_list)}_COMB.zarr')
        TIMESTAMP_DIR = os.path.join(Test_X_DIR, f'TIMESTAMP_{year_list_to_str(self.test_year_list)}_COMB.npy')
        
        
        MASK_zarr_DIR = os.path.join(Test_X_DIR, f'hmiMetadata_2_year_{year_list_to_str(self.test_year_list)}_COMB.zarr')
        
        
        if os.path.exists(VISUALIZE_PATH):
            print()
            print(f'VISUALIZATION: {VISUALIZE_PATH} already exists')
            print(f'Enter yes to continue')
            answ = input()
            if str(answ) != 'yes':
                exit()
        
        if not os.path.exists(VISUALIZE_PATH):
            os.makedirs(VISUALIZE_PATH)
        
        
        if bin_class != None:
            if ('latBin' in bin_class._get_description()) and ('lonBin' in bin_class._get_description()):
                lon_lat_bin(NPY_SAVE_DIR, Test_y_ZARR_DIR, MASK_zarr_DIR, VISUALIZE_PATH, TIMESTAMP_DIR, y, t, bin_class)
                return
        
        '''
        if Pole_only_index == None:
            full_bin_plot(NPY_SAVE_DIR, TIMESTAMP_DIR, MASK_zarr_DIR, VISUALIZE_PATH, np.arange(7749), y, False)
        else:
            full_bin_plot(NPY_SAVE_DIR, TIMESTAMP_DIR, MASK_zarr_DIR, VISUALIZE_PATH, np.arange(Pole_only_index, 7749), y, True)
        '''
        
        
        
        visualize_pipeline(NPY_SAVE_DIR, Test_y_ZARR_DIR, MASK_zarr_DIR, VISUALIZE_PATH, TIMESTAMP_DIR, img_idx_list, y, bin_class)
        
    
        test_zarr = zarr.open(Test_y_ZARR_DIR, mode='r')
        NUM_SAMPLES = test_zarr.shape[0]
     
        eval_dict = {"NPY_SAVE_DIR": NPY_SAVE_DIR,
                     "NUM_SAMPLES": NUM_SAMPLES,
                     "Test_y_ZARR_DIR": Test_y_ZARR_DIR,
                     "MASK_zarr_DIR": MASK_zarr_DIR,
                     "bin_class": bin_class,
                     "t": t,}
        
        
        
        write_html(eval_dict, VISUALIZE_PATH, TIMESTAMP_DIR, y, Pole_only_index, bin_class)

    def visualize_prob(self, y:str, data_type:str, GLOBAL_DEVICE, model, SAVE_PATH, img_idx_list, OBS_list = None, bins=[]):
        #TAR_FILE_DIR = "/nfs/turbo/fouheyUnrep/ruoyuw/FINAL_DATASET.tar"
        #tmpdir =  tmp_data_dir.TempSSDPath(TAR_FILE_DIR, 'test_dataset', logging=True)
        
        assert(data_type == 'pipeline')
        if OBS_list == None:
            model_type = 'pipeline'
        else:
            model_type = year_list_to_str(OBS_list)

        MODEL_SAVE_PATH = os.path.join(self.NFS_MODEL_SAVE_DIR, f'{model_type}_MODEL')
        MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, y)
        
        arr = sorted(os.listdir(MODEL_SAVE_PATH))
        print()
        for i in range(len(arr)):
            print(f'{i}: {arr[i]}')
        print('Enter the index of model to load from:')
        index = int(input())
        MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, arr[index])
        print(f'Loading from {MODEL_SAVE_PATH}')
        model_name = arr[index]
        
        tmp_npy_dir = os.path.join(MODEL_SAVE_PATH, 'Train_loss.npy')
        loss_arr = np.load(tmp_npy_dir)
        print(f'The model has been trained for {len(loss_arr)} epoch')
        
        arr_list = [filename for filename in (os.listdir(MODEL_SAVE_PATH)) if filename.startswith('epoch')]
        
        for name in arr_list:
            print(name)
        
        print('Enter epoch number')
        epoch = int(input())

        model_out_path = os.path.join(MODEL_SAVE_PATH, f'epoch={epoch}.checkpoint.pth')
        print(f'Loading model for epoch {epoch} from {model_out_path}')

        model.load_state_dict(torch.load(model_out_path, map_location=GLOBAL_DEVICE)['model'])
        model.eval()
        
        Test_X_DIR = os.path.join(self.Test_data_DIR, 'Shared')
        Test_y_DIR = os.path.join(self.Test_data_DIR, f'{y}')

        Test_X_ZARR_DIR = os.path.join(Test_X_DIR, f'X_{data_type}_year_{year_list_to_str(self.test_year_list)}_COMB.zarr')
        Test_y_ZARR_DIR = os.path.join(Test_y_DIR, f'{y}_year_{year_list_to_str(self.test_year_list)}_COMB.zarr')

        DISK_MASK_DIR = os.path.join(Test_X_DIR, f'hmiMetadata_2_year_{year_list_to_str(self.test_year_list)}_COMB.zarr')
        
        Train_y_DIR = os.path.join(self.Train_data_DIR, f'{y}')
        Train_X_DIR = os.path.join(self.Train_data_DIR, 'Shared')
        X_NORM_DIR = os.path.join(Train_X_DIR, 'X_Norm_param.npy')
        y_NORM_DIR = os.path.join(Train_y_DIR, 'y_Norm_param.npy')
        
        OBS_TEST_DIR_LIST = None
        OBS_TRAIN_NORM_DIR_LIST = None
        
        if OBS_list != None:
            OBS_TEST_DIR_LIST = []
            OBS_TRAIN_NORM_DIR_LIST = []
            for OBS_NAME in OBS_list:
                OBS_TEST_DIR_LIST.append(os.path.join(Test_X_DIR, f'{OBS_NAME}_year_{year_list_to_str(self.test_year_list)}_COMB.zarr'))
                OBS_TRAIN_NORM_DIR_LIST.append(os.path.join(Train_X_DIR, f'{OBS_NAME}_Norm_param.npy'))

        
        test_dataset = HMI_Pipeline(Test_X_ZARR_DIR, Test_y_ZARR_DIR, X_NORM_DIR, y_NORM_DIR, DISK_MASK_DIR, 'test', GLOBAL_DEVICE, False, OBS_TEST_DIR_LIST, OBS_TRAIN_NORM_DIR_LIST, True)
        test_loader = DataLoader(test_dataset, batch_size=1, num_workers=8, pin_memory=False)

        SAVE_PATH = SAVE_PATH
        SAVE_PATH = os.path.join(SAVE_PATH, f'{y}')
        SAVE_PATH = os.path.join(SAVE_PATH, model_type)
        SAVE_PATH = os.path.join(SAVE_PATH, model_name)
        SAVE_PATH = os.path.join(SAVE_PATH, f'epoch={epoch}')

        
        TIMESTAMP_DIR = os.path.join(Test_X_DIR, f'TIMESTAMP_{year_list_to_str(self.test_year_list)}_COMB.npy')
        visualize_prob(test_loader, model, GLOBAL_DEVICE, SAVE_PATH, TIMESTAMP_DIR, bins, img_idx_list, y)
    
    
    def visualize_prob_val(self, y:str, data_type:str, GLOBAL_DEVICE, model, SAVE_PATH, img_idx_list, OBS_list = None, bins=[]):
        #TAR_FILE_DIR = "/nfs/turbo/fouheyUnrep/ruoyuw/FINAL_DATASET.tar"
        #tmpdir =  tmp_data_dir.TempSSDPath(TAR_FILE_DIR, 'test_dataset', logging=True)
        
        assert(data_type == 'pipeline')
        if OBS_list == None:
            model_type = 'pipeline'
        else:
            model_type = year_list_to_str(OBS_list)

        MODEL_SAVE_PATH = os.path.join(self.NFS_MODEL_SAVE_DIR, f'{model_type}_MODEL')
        MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, y)
        
        arr = sorted(os.listdir(MODEL_SAVE_PATH))
        print()
        for i in range(len(arr)):
            print(f'{i}: {arr[i]}')
        print('Enter the index of model to load from:')
        index = int(input())
        MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, arr[index])
        print(f'Loading from {MODEL_SAVE_PATH}')
        model_name = arr[index]
        
        tmp_npy_dir = os.path.join(MODEL_SAVE_PATH, 'Train_loss.npy')
        loss_arr = np.load(tmp_npy_dir)
        print(f'The model has been trained for {len(loss_arr)} epoch')
        
        arr_list = [filename for filename in (os.listdir(MODEL_SAVE_PATH)) if filename.startswith('epoch')]
        
        for name in arr_list:
            print(name)
        
        print('Enter epoch number')
        epoch = int(input())

        model_out_path = os.path.join(MODEL_SAVE_PATH, f'epoch={epoch}.checkpoint.pth')
        print(f'Loading model for epoch {epoch} from {model_out_path}')

        model.load_state_dict(torch.load(model_out_path, map_location=GLOBAL_DEVICE)['model'])
        model.eval()
        
        Val_X_DIR = os.path.join(self.VAL_data_DIR, 'Shared')
        Val_y_DIR = os.path.join(self.VAL_data_DIR, f'{y}')

        Val_X_ZARR_DIR = os.path.join(Val_X_DIR, f'X_{data_type}_year_{year_list_to_str(self.val_year_list)}_COMB.zarr')
        Val_y_ZARR_DIR = os.path.join(Val_y_DIR, f'{y}_year_{year_list_to_str(self.val_year_list)}_COMB.zarr')

        DISK_MASK_DIR = os.path.join(Val_X_DIR, f'hmiMetadata_2_year_{year_list_to_str(self.val_year_list)}_COMB.zarr')
        
        Train_y_DIR = os.path.join(self.Train_data_DIR, f'{y}')
        Train_X_DIR = os.path.join(self.Train_data_DIR, 'Shared')
        X_NORM_DIR = os.path.join(Train_X_DIR, 'X_Norm_param.npy')
        y_NORM_DIR = os.path.join(Train_y_DIR, 'y_Norm_param.npy')
        
        OBS_TEST_DIR_LIST = None
        OBS_TRAIN_NORM_DIR_LIST = None
        
        if OBS_list != None:
            OBS_TEST_DIR_LIST = []
            OBS_TRAIN_NORM_DIR_LIST = []
            for OBS_NAME in OBS_list:
                OBS_TEST_DIR_LIST.append(os.path.join(Val_X_DIR, f'{OBS_NAME}_year_{year_list_to_str(self.val_year_list)}_COMB.zarr'))
                OBS_TRAIN_NORM_DIR_LIST.append(os.path.join(Train_X_DIR, f'{OBS_NAME}_Norm_param.npy'))

        
        val_dataset = HMI_Pipeline(Val_X_ZARR_DIR, Val_y_ZARR_DIR, X_NORM_DIR, y_NORM_DIR, DISK_MASK_DIR, 'validation', GLOBAL_DEVICE, False, OBS_TEST_DIR_LIST, OBS_TRAIN_NORM_DIR_LIST, True)
        val_loader = DataLoader(val_dataset, batch_size=1, num_workers=8, pin_memory=False)

        SAVE_PATH = SAVE_PATH
        SAVE_PATH = os.path.join(SAVE_PATH, f'{y}')
        SAVE_PATH = os.path.join(SAVE_PATH, model_type)
        SAVE_PATH = os.path.join(SAVE_PATH, model_name)
        SAVE_PATH = os.path.join(SAVE_PATH, f'epoch={epoch}')

        
        TIMESTAMP_DIR = os.path.join(Val_X_DIR, f'TIMESTAMP_{year_list_to_str(self.val_year_list)}_COMB.npy')
        visualize_prob(val_loader, model, GLOBAL_DEVICE, SAVE_PATH, TIMESTAMP_DIR, bins, img_idx_list, y)
        

    def predict_val(self, y:str, data_type:str, GLOBAL_DEVICE, model, OBS_list = None, bins=[], whether_CI = False):
        assert(data_type == 'pipeline')
        if OBS_list == None:
            model_type = 'pipeline'
        else:
            model_type = year_list_to_str(OBS_list)

        MODEL_SAVE_PATH = os.path.join(self.NFS_MODEL_SAVE_DIR, f'{model_type}_MODEL')
        MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, y)
        
        arr = sorted(os.listdir(MODEL_SAVE_PATH))
        print()
        for i in range(len(arr)):
            print(f'{i}: {arr[i]}')
        print('Enter the index of model to load from:')
        index = int(input())
        MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, arr[index])
        print(f'Loading from {MODEL_SAVE_PATH}')
        model_name = arr[index]
        
        tmp_npy_dir = os.path.join(MODEL_SAVE_PATH, 'Train_loss.npy')
        loss_arr = np.load(tmp_npy_dir)
        print(f'The model has been trained for {len(loss_arr)} epoch')
        
        arr_list = [filename for filename in (os.listdir(MODEL_SAVE_PATH)) if filename.startswith('epoch')]
        
        for name in arr_list:
            print(name)
        
        print('Enter epoch number')
        epoch = int(input())

        model_out_path = os.path.join(MODEL_SAVE_PATH, f'epoch={epoch}.checkpoint.pth')
        print(f'Loading model for epoch {epoch} from {model_out_path}')

        model.load_state_dict(torch.load(model_out_path, map_location=GLOBAL_DEVICE)['model'])
        model.eval()

        Val_X_DIR = os.path.join(self.VAL_data_DIR, 'Shared')
        Val_y_DIR = os.path.join(self.VAL_data_DIR, f'{y}')

        Val_X_ZARR_DIR = os.path.join(Val_X_DIR, f'X_{data_type}_year_{year_list_to_str(self.val_year_list)}_COMB.zarr')
        Val_y_ZARR_DIR = os.path.join(Val_y_DIR, f'{y}_year_{year_list_to_str(self.val_year_list)}_COMB.zarr')

        DISK_MASK_DIR = os.path.join(Val_X_DIR, f'hmiMetadata_2_year_{year_list_to_str(self.val_year_list)}_COMB.zarr')
        
        Train_y_DIR = os.path.join(self.Train_data_DIR, f'{y}')
        Train_X_DIR = os.path.join(self.Train_data_DIR, 'Shared')
        X_NORM_DIR = os.path.join(Train_X_DIR, 'X_Norm_param.npy')
        y_NORM_DIR = os.path.join(Train_y_DIR, 'y_Norm_param.npy')
        
        OBS_TEST_DIR_LIST = None
        OBS_TRAIN_NORM_DIR_LIST = None
        
        if OBS_list != None:
            OBS_TEST_DIR_LIST = []
            OBS_TRAIN_NORM_DIR_LIST = []
            for OBS_NAME in OBS_list:
                OBS_TEST_DIR_LIST.append(os.path.join(Val_X_DIR, f'{OBS_NAME}_year_{year_list_to_str(self.val_year_list)}_COMB.zarr'))
                OBS_TRAIN_NORM_DIR_LIST.append(os.path.join(Train_X_DIR, f'{OBS_NAME}_Norm_param.npy'))

        test_dataset = HMI_Pipeline(Val_X_ZARR_DIR, None, X_NORM_DIR, None, DISK_MASK_DIR, 'validation', GLOBAL_DEVICE, True, OBS_TEST_DIR_LIST, OBS_TRAIN_NORM_DIR_LIST)
        test_loader = DataLoader(test_dataset, batch_size=1, num_workers=8, pin_memory=False)

        PRED_DIR = self.PREDICT_SAVE_DIR
        PRED_DIR = os.path.join(PRED_DIR, f'{y}')
        PRED_DIR = os.path.join(PRED_DIR, model_type)
        PRED_DIR = os.path.join(PRED_DIR, model_name)
        PRED_DIR = os.path.join(PRED_DIR, f'epoch={epoch}')
        
        TIMESTAMP_DIR = os.path.join(Val_X_DIR, f'TIMESTAMP_{year_list_to_str(self.val_year_list)}_COMB.npy')
        if bins == []:
            print('Regression')
            pred_pipeline(test_loader, model, GLOBAL_DEVICE, PRED_DIR, y_NORM_DIR, TIMESTAMP_DIR)
        else:
            classification_predict_with_noise(test_loader, model, GLOBAL_DEVICE, PRED_DIR, TIMESTAMP_DIR, bins, whether_CI)

    def visualize_val(
        self, 
        y:str, 
        t, 
        img_idx_list, 
        OBS_list = None, 
        Pole_only_index = None,
        bin_class = None
        ):
        data_type = 'pipeline'
        if OBS_list == None:
            model_type = 'pipeline'
        else:
            model_type = year_list_to_str(OBS_list)
        
        NPY_SAVE_DIR = self.PREDICT_SAVE_DIR
        NPY_SAVE_DIR = os.path.join(NPY_SAVE_DIR, f'{y}')
        NPY_SAVE_DIR = os.path.join(NPY_SAVE_DIR, model_type)
        
        # Get the npy save DIR
        if True:
            arr = sorted(os.listdir(NPY_SAVE_DIR))
            print()
            for i in range(len(arr)):
                print(f'{i}: {arr[i]}')
            print('Enter the index of model to load from:')
            #index = int(input())
            index = 0
            model_name = arr[index]
        NPY_SAVE_DIR = os.path.join(NPY_SAVE_DIR, model_name)

        if True:
            arr = sorted(os.listdir(NPY_SAVE_DIR))
            print()
            for i in range(len(arr)):
                print(f'{i}: {arr[i]}')
            print('Enter the index of epoch to load from:')
            #index = int(input())
            index = 0
            epoch = arr[index]
        else:
            epoch = f'epoch={epoch}'
        NPY_SAVE_DIR = os.path.join(NPY_SAVE_DIR, epoch)

        VISUALIZE_PATH = (self.PREDICT_SAVE_DIR).replace("PREDICT", "VISUALIZE")
        
        VISUALIZE_PATH = os.path.join(VISUALIZE_PATH, f'{y}')
        VISUALIZE_PATH = os.path.join(VISUALIZE_PATH, f'{model_name}')
        VISUALIZE_PATH = os.path.join(VISUALIZE_PATH, f'epoch={epoch}')
        
        description = 'standard'

        if bin_class != None:
            description = bin_class._get_description()
        
        VISUALIZE_PATH = os.path.join(VISUALIZE_PATH, description)
        
        Val_X_DIR = os.path.join(self.VAL_data_DIR, 'Shared')
        Val_y_DIR = os.path.join(self.VAL_data_DIR, f'{y}')
        Val_y_ZARR_DIR = os.path.join(Val_y_DIR, f'{y}_year_{year_list_to_str(self.val_year_list)}_COMB.zarr')
        TIMESTAMP_DIR = os.path.join(Val_X_DIR, f'TIMESTAMP_{year_list_to_str(self.val_year_list)}_COMB.npy')
        
        
        MASK_zarr_DIR = os.path.join(Val_X_DIR, f'hmiMetadata_2_year_{year_list_to_str(self.val_year_list)}_COMB.zarr')
        
        
        if os.path.exists(VISUALIZE_PATH):
            print()
            print(f'VISUALIZATION: {VISUALIZE_PATH} already exists')
            print(f'Enter yes to continue')
            answ = input()
            if str(answ) != 'yes':
                exit()
        
        if not os.path.exists(VISUALIZE_PATH):
            os.makedirs(VISUALIZE_PATH)
        
        
        if bin_class != None:
            if ('latBin' in bin_class._get_description()) and ('lonBin' in bin_class._get_description()):
                lon_lat_bin(NPY_SAVE_DIR, Val_y_ZARR_DIR, MASK_zarr_DIR, VISUALIZE_PATH, TIMESTAMP_DIR, y, t, bin_class)
                return
        
        '''
        if Pole_only_index == None:
            full_bin_plot(NPY_SAVE_DIR, TIMESTAMP_DIR, MASK_zarr_DIR, VISUALIZE_PATH, np.arange(7749), y, False)
        else:
            full_bin_plot(NPY_SAVE_DIR, TIMESTAMP_DIR, MASK_zarr_DIR, VISUALIZE_PATH, np.arange(Pole_only_index, 7749), y, True)
        '''
        
        
        
        visualize_pipeline(NPY_SAVE_DIR, Val_y_ZARR_DIR, MASK_zarr_DIR, VISUALIZE_PATH, TIMESTAMP_DIR, img_idx_list, y, bin_class)
        
    
        test_zarr = zarr.open(Val_y_ZARR_DIR, mode='r')
        NUM_SAMPLES = test_zarr.shape[0]
     
        eval_dict = {"NPY_SAVE_DIR": NPY_SAVE_DIR,
                     "NUM_SAMPLES": NUM_SAMPLES,
                     "Test_y_ZARR_DIR": Val_y_ZARR_DIR,
                     "MASK_zarr_DIR": MASK_zarr_DIR,
                     "bin_class": bin_class,
                     "t": t,}
        
        
        
        write_html(eval_dict, VISUALIZE_PATH, TIMESTAMP_DIR, y, Pole_only_index, bin_class)




    def predict_h_flip(self, y:str, data_type:str, GLOBAL_DEVICE, model, OBS_list = None, bins=[], whether_CI = False):
        assert(data_type == 'pipeline')
        if OBS_list == None:
            model_type = 'pipeline'
        else:
            model_type = year_list_to_str(OBS_list)

        MODEL_SAVE_PATH = os.path.join(self.NFS_MODEL_SAVE_DIR, f'{model_type}_MODEL')
        MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, y)
        
        arr = sorted(os.listdir(MODEL_SAVE_PATH))
        print()
        for i in range(len(arr)):
            print(f'{i}: {arr[i]}')
        print('Enter the index of model to load from:')
        index = int(input())
        MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, arr[index])
        print(f'Loading from {MODEL_SAVE_PATH}')
        model_name = arr[index]
        
        tmp_npy_dir = os.path.join(MODEL_SAVE_PATH, 'Train_loss.npy')
        loss_arr = np.load(tmp_npy_dir)
        print(f'The model has been trained for {len(loss_arr)} epoch')
        
        arr_list = [filename for filename in (os.listdir(MODEL_SAVE_PATH)) if filename.startswith('epoch')]
        
        for name in arr_list:
            print(name)
        
        print('Enter epoch number')
        epoch = int(input())

        model_out_path = os.path.join(MODEL_SAVE_PATH, f'epoch={epoch}.checkpoint.pth')
        print(f'Loading model for epoch {epoch} from {model_out_path}')

        model.load_state_dict(torch.load(model_out_path, map_location=GLOBAL_DEVICE)['model'])
        model.eval()

        Test_X_DIR = os.path.join(self.Test_data_DIR, 'Shared')
        Test_y_DIR = os.path.join(self.Test_data_DIR, f'{y}')

        Test_X_ZARR_DIR = os.path.join(Test_X_DIR, f'X_{data_type}_year_{year_list_to_str(self.test_year_list)}_COMB.zarr')
        Test_y_ZARR_DIR = os.path.join(Test_y_DIR, f'{y}_year_{year_list_to_str(self.test_year_list)}_COMB.zarr')

        DISK_MASK_DIR = os.path.join(Test_X_DIR, f'hmiMetadata_2_year_{year_list_to_str(self.test_year_list)}_COMB.zarr')
        
        Train_y_DIR = os.path.join(self.Train_data_DIR, f'{y}')
        Train_X_DIR = os.path.join(self.Train_data_DIR, 'Shared')
        X_NORM_DIR = os.path.join(Train_X_DIR, 'X_Norm_param.npy')
        y_NORM_DIR = os.path.join(Train_y_DIR, 'y_Norm_param.npy')
        
        OBS_TEST_DIR_LIST = None
        OBS_TRAIN_NORM_DIR_LIST = None
        
        if OBS_list != None:
            OBS_TEST_DIR_LIST = []
            OBS_TRAIN_NORM_DIR_LIST = []
            for OBS_NAME in OBS_list:
                OBS_TEST_DIR_LIST.append(os.path.join(Test_X_DIR, f'{OBS_NAME}_year_{year_list_to_str(self.test_year_list)}_COMB.zarr'))
                OBS_TRAIN_NORM_DIR_LIST.append(os.path.join(Train_X_DIR, f'{OBS_NAME}_Norm_param.npy'))

        test_dataset = HMI_Pipeline(Test_X_ZARR_DIR, None, X_NORM_DIR, None, DISK_MASK_DIR, 'test', GLOBAL_DEVICE, True, OBS_TEST_DIR_LIST, OBS_TRAIN_NORM_DIR_LIST)
        test_loader = DataLoader(test_dataset, batch_size=1, num_workers=8, pin_memory=False)

        PRED_DIR = self.PREDICT_SAVE_DIR
        PRED_DIR = os.path.join(PRED_DIR, f'{y}')
        PRED_DIR = os.path.join(PRED_DIR, model_type)
        PRED_DIR = os.path.join(PRED_DIR, model_name)
        PRED_DIR = os.path.join(PRED_DIR, f'epoch={epoch}')
        
        TIMESTAMP_DIR = os.path.join(Test_X_DIR, f'TIMESTAMP_{year_list_to_str(self.test_year_list)}_COMB.npy')
        if bins == []:
            print('Regression')
            pred_pipeline(test_loader, model, GLOBAL_DEVICE, PRED_DIR, y_NORM_DIR, TIMESTAMP_DIR)
        else:
            t_h_flip_test_classification_predict_with_noise(test_loader, model, GLOBAL_DEVICE, PRED_DIR, TIMESTAMP_DIR, bins, whether_CI, y)

        
