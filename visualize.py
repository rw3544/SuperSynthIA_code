from helpers.visualize_utils import *

if __name__ == "__main__":
    # Set all the variables
    
    # Choose from ['spDisambig_Bp', 'spDisambig_Br', 'spDisambig_Bt', 'spDisambig_Field_Azimuth_Disamb', 'spInv_aB', 'spInv_Field_Azimuth', 'spInv_Field_Inclination']
    output_name_list = ["spDisambig_Bp", "spDisambig_Br", "spDisambig_Bt", "spInv_aB", "spInv_Field_Inclination", "spDisambig_Field_Azimuth_Disamb"]
    
    # Directory of folder containing Br/Bp/Bt predictions
    PRED_DIR = './YYYYY/20year_data'
    
    # Directory where the output will be saved
    VIS_SAVE_DIR = "./YYYYY/20year_data/VIS"
    
    fits_vis_packer(output_name_list, PRED_DIR, VIS_SAVE_DIR)