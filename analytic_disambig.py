from helpers.disambig_helper import *


if __name__ == "__main__":
    # Set all the variables
    
    # Directory of folder containing aB, Inclination, Azimuth predictions
    PRED_DIR = "./YYYYY/20year_data"
    
    # Directory of folder containing input IQUV fits files
    IQUV_DATA_DIR = "./twenty_years_data"
    
    # Directory where the output will be saved
    SAVE_DIR = "./YYYYY/20year_data/Disambig"
    
    # Set True to use parallel processing
    use_parallel = True
    
    # Maximum number of parallel workers
    max_parallel_workers = 8
    
    disambig_packer(PRED_DIR, IQUV_DATA_DIR, SAVE_DIR, parallel=use_parallel, max_workers = max_parallel_workers)