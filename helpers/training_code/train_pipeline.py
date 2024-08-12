from data_preprocessing import *
from model import *
from train import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import torch
import os
from datetime import datetime

from folder_name_manage import *
from model import *


#['2011_01', '2011_02', '2012_01', '2012_02','2013_01', '2013_02','2014_01', '2014_02','2015_01', '2017_02','2018_01', '2018_02','2020_01', '2020_02','2021_01', '2021_02',]

obj = Name_Manage(
        DISK_POOL= "/nfs/turbo/fouheyUnrep/ruoyuw/HMI_DISK_DATA_POOL_FINAL_NEW",
        POLE_POOL= "/nfs/turbo/fouheyUnrep/ruoyuw/HMI_POLAR_DATA_POOL_FINAL",
        MAIN_DIR= "/nfs/turbo/fouheyUnrep/ruoyuw/FINAL_PIPELINE",  
        train_year_list= ['2011_01', '2011_02', '2012_01', '2012_02','2013_01', '2013_02','2014_01', '2014_02','2015_01', '2017_02','2018_01', '2018_02','2020_01', '2020_02','2021_01', '2021_02',],
        val_year_list= ['2015_01', '2017_02'],
        test_year_list= ['2016_01', '2016_02'],
)

# MAIN_DIR= "/tmpssd/ruoyuw/test_dataset/test_PIPELINE", 
# MAIN_DIR= "/nfs/turbo/fouheyUnrep/ruoyuw/test_PIPELINE", 


GLOBAL_DEVICE = 'cuda:0'

model = UNet(24 , 1, batchnorm=False, dropout=0.3, regression=True, bins=80, bc=64).to(GLOBAL_DEVICE)
lr=1e-4
weight_decay=1e-4
eps=1e-3
#obj.build_train_dataset('spDisambig_Br', 'pipeline')
#obj.train('spDisambig_Br', 'pipeline', GLOBAL_DEVICE, lr, weight_decay, eps, model)





y_list = ['spDisambig_Bp', 'spDisambig_Br', 'spDisambig_Bt', 'spDisambig_Field_Azimuth_Disamb', 'spInv_aB', 'spInv_ChiSq_Total', 'spInv_Continuum_Intensity',
        'spInv_Damping', 'spInv_Doppler_Shift1', 'spInv_Doppler_Shift2', 'spInv_Doppler_Width', 'spInv_Field_Azimuth', 'spInv_Field_Inclination', 'spInv_Field_Strength', 'spInv_Line_Strength', 'spInv_Macro_Turbulence', 'spInv_Original_Continuum_Intensity',
        'spInv_Polarization', 'spInv_Source_Function', 'spInv_StokesV_Magnitude', 'spInv_Stray_Light_Fill_Factor', 'spInv_Stray_Light_Shift']
for y in y_list:
        obj.train(y, 'pipeline', GLOBAL_DEVICE, lr, weight_decay, eps, model,)
#obj.train('spDisambig_Bp', 'pipeline', GLOBAL_DEVICE, lr, weight_decay, eps, model,)
#obj.train('spDisambig_Bp', 'pipeline', GLOBAL_DEVICE, lr, weight_decay, eps, model,)
#obj.predict('spDisambig_Br', 'pipeline', GLOBAL_DEVICE, model)
