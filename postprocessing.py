from helpers.packing_utils import *

if __name__ == "__main__":
    # Set all the variables
    
    # Choose from ['none', 'generate_mask', 'fix_hole_overwrite', 'fix_hole_save_to_new']
    postprocessing_mode = 'fix_hole_save_to_new'
    
    # Directory of folder containing Br/Bp/Bt predictions
    PRED_DIR = './test/demo_output'
    
    # Directory of folder containing input IQUV fits files
    IQUV_DATA_DIR = './SMALL_DATASET'
    
    # Set True to use parallel processing
    use_parallel = True
    
    # Maximum number of parallel workers
    max_parallel_workers = 8
    
    # Set True to skip mask generation (You have already run code to generate masks)
    # !!! Set False in all other cases
    skip_mask_generation = False
    
    postprocessing_packer(postprocessing_mode, PRED_DIR, IQUV_DATA_DIR, use_parallel, max_parallel_workers, skip_mask_generation)