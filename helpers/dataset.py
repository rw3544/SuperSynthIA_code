import torch 
import os 
import random
import numpy as np
import zarr


from helpers.utils import *

DEBUG_MODE = False

# For debug mode, set manual seed
if DEBUG_MODE == True:
    RANDOM_SEED = 15
    print(f'For debug mode, seed = {RANDOM_SEED}')
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)



import time
# Works for pipeline and extra_layer
class HMI_Pipeline(torch.utils.data.Dataset):
    # Dataset for HMI_STOKES -> HMI_INCLINATION_INVERSION

    def __init__(
        self,
        X_DIR: str,
        y_DIR: str,
        X_norm_DIR: str,
        y_norm_DIR: str,
        disk_mask_DIR: str,
        partition: str,
        GLOBAL_DEVICE: str,
        IGNORE_Y = False,
        OBS_DIR_LIST = None,
        OBS_NORM_DIR_LIST = None,
        CLASSIFICATION = False,
        ):
        
    
        ########
        # Args:
        # 
        # Self. :
        #   partition (str): "train", "val", "test"
        #   X, y: Dataset, X is stored as img and y is stored as mask
        #   DATASET_DIR (str): Dataset Directory for preprocessed zarr 
        #   
        #   OBS_ZARR: [zarr1, zarr2, ...]
        #
        #   If CLASSIFICATION = True, no normalize
        #   
        ########
        super().__init__()

        if partition not in ["train", 'validation', "test"]:
            raise ValueError("Partition {} does not exist".format(partition))
        if OBS_DIR_LIST != None:
            assert(len(OBS_DIR_LIST) == len(OBS_NORM_DIR_LIST))

        self.X_DIR = X_DIR
        self.y_DIR = y_DIR
        self.IGNORE_Y = IGNORE_Y
        
        
        # Ensure to use the correct file
        assert('hmiMetadata_2' in disk_mask_DIR)
        self.CLASSIFICATION = CLASSIFICATION
        if partition == 'train':
            assert('TRAIN' in self.X_DIR)
            if IGNORE_Y == False:
                assert('TRAIN' in self.y_DIR)
            assert('TRAIN' in disk_mask_DIR)
        elif partition == 'validation':
            assert('VAL' in self.X_DIR)
            if IGNORE_Y == False:
                assert('VAL' in self.y_DIR)
            assert('VAL' in disk_mask_DIR)
        elif partition == 'test':
            assert('TEST' in self.X_DIR)
            if IGNORE_Y == False:
                assert('TEST' in self.y_DIR)
            assert('TEST' in disk_mask_DIR)
        
        
        self.partition = partition
        self.GLOBAL_DEVICE = GLOBAL_DEVICE
        
        
        self.X, self.y = self._load_data()       
        
        self.X_channel_size = 24
        if OBS_DIR_LIST != None:
            self.OBS_ZARR = []
            for obs_zarr_dir in OBS_DIR_LIST:
                self.OBS_ZARR.append(zarr.open(obs_zarr_dir, mode='r'))
                self.X_channel_size += 1
        
        # Get Normalization Constants
        X_norm_param = torch.from_numpy(np.load(X_norm_DIR).astype(np.float64))
        if IGNORE_Y == False:
            y_norm_param = torch.from_numpy(np.load(y_norm_DIR).astype(np.float64))
            self.y_mean = y_norm_param[0,:]
            self.y_std = y_norm_param[1, :]
        self.X_mean = X_norm_param[0, :]
        self.X_std = X_norm_param[1,:]
        
        # Load disk mask from HMIMETADATA_2 
        self.Disk_mask = zarr.open(disk_mask_DIR, mode='r')
        
        
        if OBS_DIR_LIST != None:
            # Load NORM_PARAM and stack to mean/std
            COMB_MEAN = np.zeros(self.X_channel_size)
            COMB_STD = np.zeros_like(COMB_MEAN)
            COMB_MEAN[:24] = self.X_mean
            COMB_STD[:24] = self.X_std
            for i in range(len(OBS_NORM_DIR_LIST)):
                obs_norm_dir = OBS_NORM_DIR_LIST[i]
                tmp_norm = np.load(obs_norm_dir)
                COMB_MEAN[24+i] = tmp_norm[0,:]
                COMB_STD[24+i] = tmp_norm[1,:]
            
            self.X_mean = COMB_MEAN
            self.X_std = COMB_STD
        
        # Make sure the shape is correct
        assert(self.X_mean.shape[0] == self.X_channel_size)
        assert(self.X_std.shape[0] == self.X_channel_size)
        if IGNORE_Y == False:
            assert(self.y_mean.shape[0] == self.y.shape[1])
            assert(self.y_std.shape[0] == self.y.shape[1])
        


    def _load_data(
        self,
        ):
        print(f'loading {self.partition} data ...')
        if self.IGNORE_Y == False:
            return zarr.open(self.X_DIR, mode='r'), zarr.open(self.y_DIR, mode='r')
        else:
            return zarr.open(self.X_DIR, mode='r'), None


    def __len__(self):
        if self.IGNORE_Y == False:
            if self.X.shape[0] != self.y.shape[0]:
                raise ValueError(f'Unequal X,y length, len(X)={self.X.shape[0]}, len(y)={self.y.shape[0]}')
        return self.X.shape[0]

    def _normalize(self, X, y):
        if self.IGNORE_Y == False:
            return (X-self.X_mean)/self.X_std, (y-self.y_mean)/self.y_std
        else:
            return (X-self.X_mean)/self.X_std, None
    
    def __getitem__(self, idx):
        x_item = self.X[idx]
        
        if self.IGNORE_Y == False:
            y_item = self.y[idx]
        
        dmask_item = self.Disk_mask[idx]
                

        # X,y originally in N,C,H,W
        X = (torch.from_numpy(x_item).float())
        X = X.permute(1,2,0)
        
        y = None
        if self.IGNORE_Y == False:
            y = (torch.from_numpy(y_item).float())
            y = y.permute(1,2,0)
            
        dmask = (torch.from_numpy(dmask_item).float())
        dmask = dmask.permute(1,2,0)
        
        
        # X,y in H,W,C
        X = de_pad(X)
        if self.IGNORE_Y == False:
            y = de_pad(y)
        dmask = de_pad(dmask)
        
        
        # add OBS_V to X
        if self.X_channel_size != 24:
            H,W,C = X.shape
            tmp_X = torch.zeros((H, W, self.X_channel_size))
            tmp_X[:,:,:C] = X
            for i in range(len(self.OBS_ZARR)):
                
                tmp_X[:,:,C+i] = torch.ones((H,W))*((self.OBS_ZARR[i])[idx])
            
            tmp_X = tmp_X.float()
            X = tmp_X    
                
        
        # Normalize
        if self.CLASSIFICATION == False:
            X,y = self._normalize(X,y)
        else:
            X,_ = self._normalize(X,y)
        
        H,W,C = X.shape
        X = X.reshape(1,H,W,C).permute(0,3,1,2)
        if self.IGNORE_Y == False:
            y = y.reshape(1,1,H,W)
        dmask = dmask.reshape(1,1,H,W).bool()

        
        x_mask = ~torch.isnan(X) # remove nans
        if self.IGNORE_Y == False:
            y_mask = ~torch.isnan(y) # remove nans


        comb_mask = dmask.clone()

        
        if self.IGNORE_Y == False:
            comb_mask *= y_mask

        
        for i in range(x_mask.shape[1]):
            comb_mask = comb_mask*x_mask[:,i,:,:]

        #X,y in N,C,H,W
        if self.IGNORE_Y == False:
            return {'X': X,
                    'y': y,
                    'mask': comb_mask,
                    'disk_mask': dmask}
        else:
            return {'X': X,
                    'mask': comb_mask,
                    'disk_mask': dmask}


# Works for extra_FC
class HMI_extra_FC(torch.utils.data.Dataset):
    # Dataset for HMI_STOKES -> HMI_INCLINATION_INVERSION

    def __init__(
        self,
        X_DIR: str,
        y_DIR: str,
        X_norm_DIR: str,
        y_norm_DIR: str,
        disk_mask_DIR: str,
        OBS_DIR_LIST: list,
        partition: str,
        GLOBAL_DEVICE: str,
        QUIET_MODE = True,
        ):
    
        ########
        # Args:
        # 
        # Self. :
        #   partition (str): "train", "val", "test"
        #   X, y: Dataset, X is stored as img and y is stored as mask
        #   DATASET_DIR (str): Dataset Directory for preprocessed zarr 
        #   
        #   
        ########
        super().__init__()

        if partition not in ["train", 'validation', "test"]:
            raise ValueError("Partition {} does not exist".format(partition))

        self.X_DIR = X_DIR
        self.y_DIR = y_DIR
        # Ensure to use the correct file
        assert('hmiMetadata_2' in disk_mask_DIR)
        if partition == 'train':
            assert('TRAIN' in self.X_DIR)
            assert('TRAIN' in self.y_DIR)
            assert('TRAIN' in disk_mask_DIR)
        elif partition == 'validation':
            assert('VAL' in self.X_DIR)
            assert('VAL' in self.y_DIR)
            assert('VAL' in disk_mask_DIR)
        elif partition == 'test':
            assert('TEST' in self.X_DIR)
            assert('TEST' in self.y_DIR)
            assert('TEST' in disk_mask_DIR)
        if QUIET_MODE == False:
            print(f'Loading X_{partition} from {self.X_DIR}')
            print(f'Loading y_{partition} from {self.y_DIR}')
        
        self.partition = partition
        self.GLOBAL_DEVICE = GLOBAL_DEVICE
        
        self.X, self.y = self._load_data()
        
        # Get Normalization Constants
        X_norm_param = torch.from_numpy(np.load(X_norm_DIR).astype(np.float64))
        y_norm_param = torch.from_numpy(np.load(y_norm_DIR).astype(np.float64))
        self.X_mean = X_norm_param[0, :]
        self.X_std = X_norm_param[1,:]
        self.y_mean = y_norm_param[0,:]
        self.y_std = y_norm_param[1, :]
        
        # Load disk mask from HMIMETADATA_2 
        self.Disk_mask = zarr.open(disk_mask_DIR, mode='r')
        
        #self.
        #for obs_dir in OBS_DIR_LIST:
            
        
        # Make sure the shape is correct
        assert(self.X_mean.shape[0] == self.X.shape[1])
        assert(self.X_std.shape[0] == self.X.shape[1])
        assert(self.y_mean.shape[0] == self.y.shape[1])
        assert(self.y_std.shape[0] == self.y.shape[1])
        


    def _load_data(
        self,
        ):
        print(f'loading {self.partition} data ...')
        return zarr.open(self.X_DIR, mode='r'), zarr.open(self.y_DIR, mode='r')


    def __len__(self):
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(f'Unequal X,y length, len(X)={self.X.shape[0]}, len(y)={self.y.shape[0]}')
        return self.X.shape[0]

    def _normalize(self, X, y):
        return (X-self.X_mean)/self.X_std, (y-self.y_mean)/self.y_std
    
    def __getitem__(self, idx):
        x_item = self.X[idx]
        
        y_item = self.y[idx]
        
        dmask_item = self.Disk_mask[idx]
        
        # X,y originally in N,C,H,W
        X = (torch.from_numpy(x_item).float())
        X = X.permute(1,2,0)
        y = (torch.from_numpy(y_item).float())
        y = y.permute(1,2,0)
        dmask = (torch.from_numpy(dmask_item).float())
        dmask = dmask.permute(1,2,0)
        
        
        # X,y in H,W,C
        X = de_pad(X)
        y = de_pad(y)
        dmask = de_pad(dmask)

        
        X,y = self._normalize(X,y)
        
        H,W,C = X.shape
        X = X.reshape(1,H,W,C).permute(0,3,1,2)
        y = y.reshape(1,1,H,W)
        dmask = dmask.reshape(1,1,H,W).bool()

        
        x_mask = ~torch.isnan(X) # remove nans
        y_mask = ~torch.isnan(y) # remove nans
        comb_mask = y_mask*dmask

        
        for i in range(x_mask.shape[1]):
            comb_mask = comb_mask*x_mask[:,i,:,:]
            
        
        #X,y in N,C,H,W
        return {'X': X,
                'y': y,
                'mask': comb_mask,
                'disk_mask': dmask}


    

