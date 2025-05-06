import os
import numpy as np
import sunpy.map, sunpy.io
import astropy.io.fits as fits
from helpers.utils import isLocked
import helpers.disambiguation_utils as disambiguation
import concurrent.futures
from helpers.utils import pack_to_fits, get_data_from_fits


def disambig_packer(BASE_PRED_DIR, IQUV_DIR, SAVE_DIR, parallel=False, max_workers = 8):
    outputs = ["spDisambig_Bp","spDisambig_Br","spDisambig_Bt"]

    for o in outputs:
        if not os.path.exists(os.path.join(SAVE_DIR, o)):
            os.makedirs(os.path.join(SAVE_DIR, o))
    
    fieldBase = os.path.join(BASE_PRED_DIR, 'spInv_aB')
    inclinationBase = os.path.join(BASE_PRED_DIR, 'spInv_Field_Inclination')
    azimuthDisBase = os.path.join(BASE_PRED_DIR, 'spDisambig_Field_Azimuth_Disamb')
    
    fieldFiles = sorted([f for f in os.listdir(fieldBase) if f.endswith('.aB.fits')])
    inclinationFiles = sorted([f for f in os.listdir(inclinationBase) if f.endswith('.Field_Inclination.fits')])
    azimuthDisFiles = sorted([f for f in os.listdir(azimuthDisBase) if f.endswith('.Field_Azimuth_Disamb.fits')])
    I0Files = sorted([fn for fn in os.listdir(IQUV_DIR) if fn.endswith("I0.fits")])

    timestamps = [fn.split(".")[2] for fn in fieldFiles]
    
    if parallel == False:
        for tsi, ts in enumerate(timestamps):
            print(tsi, len(timestamps))
            disambig_sample(ts, fieldFiles, inclinationFiles, azimuthDisFiles, I0Files, fieldBase, inclinationBase, azimuthDisBase, IQUV_DIR, SAVE_DIR)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers = max_workers) as executor:
            tasks = [(ts, fieldFiles, inclinationFiles, azimuthDisFiles, I0Files, fieldBase, inclinationBase, azimuthDisBase, IQUV_DIR, SAVE_DIR) for ts in timestamps]
            # Use executor.map with the wrapper function
            results = executor.map(disambig_sample_wrapper, tasks)
            # Consume the iterator to wait for tasks to complete
            for _ in results:
                pass
        

# Define the wrapper at the module level
def disambig_sample_wrapper(args):
    return disambig_sample(*args)       

def disambig_sample(ts, fieldFiles, inclinationFiles, azimuthDisFiles, I0Files, fieldBase, inclinationBase, azimuthDisBase, I0Base, target):
    fieldFn = [fn for fn in fieldFiles if fn.split(".")[2] == ts][0]

    outputFile = os.path.join(target, "spDisambig_Br", fieldFn)
    if os.path.exists(outputFile) or isLocked(outputFile+".lock"):
        return

    inclinationFn = [fn for fn in inclinationFiles if fn.split(".")[2] == ts][0]
    azimuthDisFn = [fn for fn in azimuthDisFiles if fn.split(".")[2] == ts][0]
    I0Fn = [fn for fn in I0Files if fn.split(".")[2] == ts][0]

    field = get_data_from_fits(os.path.join(fieldBase, fieldFn))
    inclination = get_data_from_fits(os.path.join(inclinationBase, inclinationFn))
    azimuthDisambig = get_data_from_fits(os.path.join(azimuthDisBase, azimuthDisFn))
    azimuthDisambig = azimuthDisambig+90
    

    fitsForHeader = fits.open(os.path.join(I0Base, I0Fn))
    sunpymap = sunpy.map.Map(os.path.join(I0Base, I0Fn))

    C = disambiguation.CoordinateTransformMapPlusDisambigArrays(azimuthDisambig, field, inclination, fitsForHeader[1].header, sunpymap)
    latlon, bptr = disambiguation.CoordinateTransformMapPlusDisambigArrays.ccd(C)

    bp, bt, br = bptr[:,:,0], bptr[:,:,1], bptr[:,:,2]
    #bp, bt, br = bptr[::-1,::-1,0], bptr[::-1,::-1,1], bptr[::-1,::-1,2]
    #bt = -bt
    
    pack_to_fits(os.path.join(target, "spDisambig_Bp"), I0Fn, bp, fitsForHeader, 'spDisambig_Bp', '_disambiged', whether_flip=False)
    pack_to_fits(os.path.join(target, "spDisambig_Br"), I0Fn, br, fitsForHeader, 'spDisambig_Br', '_disambiged', whether_flip=False)
    pack_to_fits(os.path.join(target, "spDisambig_Bt"), I0Fn, bt, fitsForHeader, 'spDisambig_Bt', '_disambiged', whether_flip=False)
    
    #np.save(os.path.join(target, "Bp", fieldFn.replace('aB', 'Bp')), bp.astype(np.float32))
    #np.save(os.path.join(target, "Bt", fieldFn.replace('aB', 'Bt')), bt.astype(np.float32))
    #np.save(os.path.join(target, "Br", fieldFn.replace('aB', 'Br')), br.astype(np.float32))

    os.rmdir(outputFile+".lock")

