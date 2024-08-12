from math import nan
import os
from data_preprocessing import *
import sys, argparse
# Combine zarr in Pool to build training dataset



from folder_name_manage import *
from model import *


#['2011_01', '2011_02', '2012_01', '2012_02','2013_01', '2013_02','2014_01', '2014_02','2015_01', '2017_02','2018_01', '2018_02','2020_01', '2020_02','2021_01', '2021_02',]
'''
obj = Name_Manage(
        DISK_POOL= "/nfs/turbo/fouheyUnrep/ruoyuw/HMI_DISK_DATA_POOL_FINAL_NEW",
        POLE_POOL= "/nfs/turbo/fouheyUnrep/ruoyuw/HMI_POLAR_DATA_POOL_FINAL",
        MAIN_DIR= "/nfs/turbo/fouheyUnrep/ruoyuw/FINAL_PIPELINE", 
        NFS_MODEL_SAVE_DIR= "/nfs/turbo/fouheyUnrep/ruoyuw/MODEL",
        PREDICT_SAVE_DIR="/nfs/turbo/fouheyUnrep/ruoyuw/PREDICT"
        train_year_list= ['2011_01', '2011_02', '2012_01', '2012_02','2013_01', '2013_02','2014_01', '2014_02','2015_01', '2017_02','2018_01', '2018_02','2020_01', '2020_02','2021_01', '2021_02',],
        val_year_list= ['2015_01', '2017_02'],
        test_year_list= ['2016_01', '2016_02'],
)
'''



p = argparse.ArgumentParser()
p.add_argument('--device', default='cuda:0', type=str, help='cuda GPU to run the network on')
#p.add_argument('-y', '--yname', required=True, type=str, help='Output to train')
args = p.parse_args()


def main():
        #yname = args.yname
        #y_list = ['spDisambig_Bp', 'spDisambig_Br', 'spDisambig_Bt', 'spDisambig_Field_Azimuth_Disamb', 'spInv_aB', 'spInv_ChiSq_Total', 'spInv_Continuum_Intensity',
        #        'spInv_Damping', 'spInv_Doppler_Shift1', 'spInv_Doppler_Shift2', 'spInv_Doppler_Width', 'spInv_Field_Azimuth', 'spInv_Field_Inclination', 'spInv_Field_Strength', 'spInv_Line_Strength', 'spInv_Macro_Turbulence', 'spInv_Original_Continuum_Intensity',
        #        'spInv_Polarization', 'spInv_Source_Function', 'spInv_StokesV_Magnitude', 'spInv_Stray_Light_Fill_Factor', 'spInv_Stray_Light_Shift']
        y_list = ['spDisambig_Bp', 'spDisambig_Br', 'spDisambig_Bt', 'spDisambig_Field_Azimuth_Disamb', 'spInv_aB', 
                 'spInv_Field_Azimuth', 'spInv_Field_Inclination', 'spInv_Field_Strength', ]
        #assert(yname in y_list)
        GLOBAL_DEVICE = args.device
       
        
        obj = Name_Manage(
                DISK_POOL= "/nfs/fouhey_pool3/users/ruoyuw/HMI_DISK_DATA_POOL_FINAL_2023",
                POLE_POOL= "/nfs/fouhey_pool3/users/ruoyuw/HMI_POLAR_DATA_POOL_FINAL_2023",
                MAIN_DIR= "/nfs/fouhey_pool3/users/ruoyuw/Standard_Split_test_2016_Dataset_2023", 
                NFS_MODEL_SAVE_DIR= "/nfs/turbo/fouheyUnrep/ruoyuw/MODEL",
                PREDICT_SAVE_DIR= "/nfs/turbo/fouheyUnrep/ruoyuw/PREDICT",
                train_year_list= ['2011_01', '2011_02', '2012_01', '2012_02','2013_01', '2013_02','2014_01', '2014_02','2018_01', '2018_02', '2019_01', '2019_02', '2020_01', '2020_02','2021_01', '2021_02',],
                val_year_list= ['2015_01', '2017_02'],
                test_year_list= ['2016_01', '2016_02'],
        )



        #OBS_list= ['hmiHeader_OBS_VN', 'hmiHeader_OBS_VR', 'hmiHeader_OBS_VW']
        #OBS_list= ['hmiHeader_OBS_VN', 'hmiHeader_OBS_VR', 'hmiHeader_OBS_VW']
        #model = UNet(27 , 1, batchnorm=False, dropout=0.3, regression=True, bins=80, bc=64).to(GLOBAL_DEVICE)
        #model = nn.DataParallel(model)
        #lr=1e-4
        #weight_decay=1e-4
        #eps=1e-3
        for yname in y_list:
                obj.build_train_dataset(yname, 'pipeline')
        #obj.train('spDisambig_Br', 'pipeline', GLOBAL_DEVICE, lr, weight_decay, eps, model)
        
        
        #obj.train(yname, 'pipeline', GLOBAL_DEVICE, lr, weight_decay, eps, model, OBS_list= ['hmiHeader_OBS_VN', 'hmiHeader_OBS_VR', 'hmiHeader_OBS_VW'])
        
        
        
        #obj.train(yname, 'pipeline', GLOBAL_DEVICE, lr, weight_decay, eps, model, OBS_list= OBS_list)
                
        
        #obj.predict('spDisambig_Br', 'pipeline', GLOBAL_DEVICE, model)

if __name__ == "__main__":
        main()