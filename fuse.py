from helpers.fuse_utils import *

if __name__ == "__main__":

    # Directory of folder containing direct SuperSynthIA Br/Bp/Bt predictions
    DIRECT_PRED_DIR = "./test/demo_output"
    
    # Directory of folder containing analytic disambiguated SuperSynthIA Br/Bp/Bt predictions
    ANALYTIC_DISAMBIG_PRED_DIR = "./test/demo_output/Disambig"

    # Directory of folder containing input IQUV fits files
    IQUV_DATA_DIR = "./SMALL_DATASET"
    
    # Directory where the output will be saved
    SAVE_DIR = "./test/demo_output/Fuse_test"
    
    # Set True to use parallel processing
    use_parallel = True
    
    # Maximum number of parallel workers
    max_parallel_workers = 8

    process_fuse_packer(DIRECT_PRED_DIR, ANALYTIC_DISAMBIG_PRED_DIR, IQUV_DATA_DIR, SAVE_DIR, parallel=use_parallel, max_parallel_workers = max_parallel_workers)
    