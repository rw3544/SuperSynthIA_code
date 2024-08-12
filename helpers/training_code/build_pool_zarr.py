from data_preprocessing import *
import os 




'''
srcBase = "/x/fouhey/SynodeVolumesPoleFinal"
srcFlowBase = "/x/fouhey/SynodeVolumesPoleFinalFlow"
targetBase = "/z/ruoyuw/test/"

year_list = ['2011', '2012', '2013', '2014', '2015']
c_year = '2012_01'

preprocess_Data(
    srcBase,
    srcFlowBase,
    targetBase,
    c_year,
    DEBUG_MODE = False,
    QUIET_MODE = False,
    ZARR_VERIFY = True
    )


c_year = '2012_02'

preprocess_Data(
    srcBase,
    srcFlowBase,
    targetBase,
    c_year,
    DEBUG_MODE = False,
    QUIET_MODE = False,
    ZARR_VERIFY = True
    )

    
'''



# Pole data
srcBase = "/x/fouhey/SynodeVolumesPoleFinal"
srcFlowBase = "/x/fouhey/SynodeVolumesPoleFinalFlow"
srcFlowDisambigBase = "/Pool3/users/fouhey/repoint/SynodeVolumesPoleRepoint/"
targetBase = "/Pool3/users/ruoyuw/SYIA_CHUNK_DATA"

year_list = ['2011_01', '2011_02', '2012_01', '2012_02','2013_01', '2013_02','2014_01', '2014_02','2015_01', '2015_02','2016_01', '2016_02','2017_01', '2017_02','2018_01', '2018_02','2019_01', '2019_02','2020_01', '2020_02','2021_01', '2021_02',]
#c_year = year_list[0]

for c_year in year_list:
    preprocess_Data(
        srcBase,
        srcFlowBase,
        srcFlowDisambigBase,
        targetBase,
        c_year,
        DEBUG_MODE = False,
        QUIET_MODE = False,
        ZARR_VERIFY = True
        )



# Disk data
srcBase = "/z/fouhey/SynodeVolumesFinal/"
srcFlowBase = "/z/fouhey/SynodeVolumesFinalFlowJune/"
srcFlowDisambigBase = "/Pool3/users/fouhey/repoint/SynodeVolumesRepoint/"
targetBase = "/Pool3/users/ruoyuw/SYIA_CHUNK_DATA"

year_list = ['2011_01', '2011_02', '2012_01', '2012_02','2013_01', '2013_02','2014_01', '2014_02','2015_01', '2015_02','2016_01', '2016_02','2017_01', '2017_02','2018_01', '2018_02','2019_01', '2019_02','2020_01', '2020_02','2021_01', '2021_02',]
#c_year = year_list[0]

for c_year in year_list:
    preprocess_Data(
        srcBase,
        srcFlowBase,
        srcFlowDisambigBase,
        targetBase,
        c_year,
        DEBUG_MODE = False,
        QUIET_MODE = False,
        ZARR_VERIFY = True
        )
'''

srcBase = "/z/fouhey/SynodeVolumesFinal/"
srcFlowBase = "/z/fouhey/SynodeVolumesFinalFlowJune/"
targetBase = "/Pool3/users/ruoyuw/HMI_DISK_DATA_POOL_FINAL_NEW"

year_list = ['2016_01', '2016_02',]
#c_year = year_list[0]

for c_year in year_list:
    rlh_preprocess_Data(
        srcBase,
        srcFlowBase,
        targetBase,
        c_year,
        DEBUG_MODE = False,
        QUIET_MODE = False,
        ZARR_VERIFY = True
        )
'''