from helpers.disambig_helper import *


if __name__ == "__main__":
    # Set all the variables
    
    # Directory of folder containing aB, Inclination, Azimuth predictions
    PRED_DIR = "./test/demo_output"
    
    # Directory of folder containing input IQUV fits files
    IQUV_DATA_DIR = "./SMALL_DATASET"
    
    # Directory where the output will be saved
    SAVE_DIR = "./test/demo_output/Disambig"
    
    # Set True to use parallel processing
    use_parallel = True
    
    # Maximum number of parallel workers
    max_parallel_workers = 8
    
    disambig_packer(PRED_DIR, IQUV_DATA_DIR, SAVE_DIR, parallel=use_parallel, max_workers = max_parallel_workers)