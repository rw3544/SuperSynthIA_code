import matplotlib.pyplot as plt
import sunpy.io 
import scipy.ndimage as ndimage
import astropy.io.fits as fits
import multiprocessing
import numpy as np
import os
from helpers.full_disk_utils import pack_to_fits


def downBin(X,k):
    h, w = X.shape
    X2 = np.zeros((h//k,w//k))
    for i in range(k):
        for j in range(k):
            X2 = X2 + X[i::k,j::k]
    return X2/(k*k)

def alphaBlend(X,Y,M,boundaryWidth):
    M2 = ndimage.binary_dilation(M,np.ones((boundaryWidth,boundaryWidth)))
    W = ndimage.gaussian_filter(M2.astype(np.float32), sigma=boundaryWidth/6)
    return X*W+Y*(1-W)



def dtBlend(X,Y,M,boundaryWidth):
#    M2 = ndimage.binary_dilation(M,np.ones((boundaryWidth,boundaryWidth)))
    D = ndimage.distance_transform_edt(M, sampling=8)
    W = np.exp(-D/boundaryWidth)
    return Y*W+X*(1-W)


def handle(tsUse, synthIAOut):
    YBrFn = [fn for fn in os.listdir(synthIAOut["Br"]) if fn.find(tsUse) != -1][0]
    YBpFn = [fn for fn in os.listdir(synthIAOut["Bp"]) if fn.find(tsUse) != -1][0]
    YBtFn = [fn for fn in os.listdir(synthIAOut["Bt"]) if fn.find(tsUse) != -1][0]

    YBrDFn = [fn for fn in os.listdir(synthIAOut["BrD"]) if fn.find(tsUse) != -1][0]
    YBpDFn = [fn for fn in os.listdir(synthIAOut["BpD"]) if fn.find(tsUse) != -1][0]
    YBtDFn = [fn for fn in os.listdir(synthIAOut["BtD"]) if fn.find(tsUse) != -1][0]


    YBr = (fits.open(os.path.join(synthIAOut["Br"],YBrFn)))[1].data
    YBp = (fits.open(os.path.join(synthIAOut["Bp"],YBpFn)))[1].data
    YBt = (fits.open(os.path.join(synthIAOut["Bt"],YBtFn)))[1].data

    YBrD = (fits.open(os.path.join(synthIAOut["BrD"],YBrDFn)))[1].data
    YBpD = (fits.open(os.path.join(synthIAOut["BpD"],YBpDFn)))[1].data
    YBtD = (fits.open(os.path.join(synthIAOut["BtD"],YBtDFn)))[1].data
    

    #YaBD = (YBrD**2+YBpD**2+YBtD**2)**0.5
    #dist = ((YBr-YBrD)**2 + (YBp-YBpD)**2 + (YBt-YBtD)**2)**0.5

    
    offDiskMask = np.isnan(YBr)
    sliverDisk = ndimage.binary_dilation(offDiskMask,np.ones((64,64)))

    updateMask = np.ones((4096,4096))
    updateMask[:400] = 0
    updateMask[-400:] = 0
    updateMask[sliverDisk] = 0

    YBrM = dtBlend(YBrD, YBr, updateMask, 300)
    YBpM = dtBlend(YBpD, YBp, updateMask, 300)
    YBtM = dtBlend(YBtD, YBt, updateMask, 300)

    targetBase = synthIAOut["targetBase"]
    IQUV_DATA_DIR = synthIAOut["IQUV_DATA_DIR"]
    I0Files = sorted([fn for fn in os.listdir(IQUV_DATA_DIR) if fn.endswith("I0.fits")])
    I0Fn = [fn for fn in I0Files if fn.split(".")[2] == tsUse][0]
    fitsForHeader = fits.open(os.path.join(IQUV_DATA_DIR, I0Fn))
    
    
    pack_to_fits(os.path.join(targetBase,"spDisambig_Br"), I0Fn, YBrM.astype(np.float32), fitsForHeader, 'spDisambig_Br', '_fused', whether_flip=False)
    pack_to_fits(os.path.join(targetBase,"spDisambig_Bp"), I0Fn, YBpM.astype(np.float32), fitsForHeader, 'spDisambig_Bp', '_fused', whether_flip=False)
    pack_to_fits(os.path.join(targetBase,"spDisambig_Bt"), I0Fn, YBtM.astype(np.float32), fitsForHeader, 'spDisambig_Bt', '_fused', whether_flip=False)

    #np.save(os.path.join(targetBase,"Br",YBrFn), YBrM.astype(np.float32))
    #np.save(os.path.join(targetBase,"Bp",YBpFn), YBpM.astype(np.float32))
    #np.save(os.path.join(targetBase,"Bt",YBtFn), YBtM.astype(np.float32))



def process_fuse_packer(DIRECT_PRED_DIR, ANALYTIC_DISAMBIG_PRED_DIR, IQUV_DATA_DIR, targetBase, parallel = True, max_parallel_workers = 8):
    if not os.path.exists(targetBase):
        os.mkdir(targetBase)
    
    synthIAOut = {}
    # Analytic
    synthIAOut["Br"] = os.path.join(ANALYTIC_DISAMBIG_PRED_DIR, "spDisambig_Br")
    synthIAOut["Bp"] = os.path.join(ANALYTIC_DISAMBIG_PRED_DIR, "spDisambig_Bp")
    synthIAOut["Bt"] = os.path.join(ANALYTIC_DISAMBIG_PRED_DIR, "spDisambig_Bt")
    
    # Direct
    synthIAOut["BrD"] = os.path.join(DIRECT_PRED_DIR, "spDisambig_Br")
    synthIAOut["BpD"] = os.path.join(DIRECT_PRED_DIR, "spDisambig_Bp")
    synthIAOut["BtD"] = os.path.join(DIRECT_PRED_DIR, "spDisambig_Bt")
    
    synthIAOut["targetBase"] = targetBase
    synthIAOut["IQUV_DATA_DIR"] = IQUV_DATA_DIR
    
    
    names = ["spDisambig_Br","spDisambig_Bp","spDisambig_Bt"]
    for n in names:
        if not os.path.exists(os.path.join(targetBase,n)):
            os.mkdir(os.path.join(targetBase,n))
    
    tss = sorted([fn.split(".")[2] for fn in os.listdir(synthIAOut["Br"])])
    
    
    if parallel:
        P = multiprocessing.Pool(processes=max_parallel_workers)
        P.starmap(handle, [(ts, synthIAOut) for ts in tss])
    else:
        for ts in tss:
            handle(ts, synthIAOut)
    







