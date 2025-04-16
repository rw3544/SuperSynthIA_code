import json, sys
from datetime import datetime
from helpers.full_disk_utils import *

def parse_json_file(file_path):
    """
    Parses a JSON file and returns its contents as a dictionary,
    ignoring keys that start with '_comment'.

    :param file_path: Path to the JSON file
    :return: Dictionary containing the JSON data
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Filter out keys starting with '_comment'
    filtered_data = {k: v for k, v in data.items() if not k.startswith('_comment')}
    return filtered_data

# For postprocessing, check if have necessary predictions: ['spDisambig_Bp', 'spDisambig_Br', 'spDisambig_Bt',]
def verify_have_necessary_preds(to_predict_list, PRED_SAVE_DIR, post_processing_mode):
    # Get all predictions that already exist
    exist_list = [name for name in os.listdir(PRED_SAVE_DIR) if os.path.isdir(os.path.join(PRED_SAVE_DIR, name))]
    
    # Decide the predictions required
    required_elements = None
    if post_processing_mode == 'none':
        return
    elif post_processing_mode == 'generate_mask':
        required_elements = ['spDisambig_Bp', 'spDisambig_Br', 'spDisambig_Bt',]
    elif post_processing_mode == 'fix_hole_overwrite' or post_processing_mode == 'fix_hole_save_to_new':
        if 'MASK' in exist_list:
            required_elements = ['spDisambig_Br']
        else:
            required_elements = ['spDisambig_Bp', 'spDisambig_Br', 'spDisambig_Bt',]
    else:
        print(f"Unknown postprocessing mode: {post_processing_mode}")
        sys.exit(1)
    
    # Check if have all necessary predictions for postprocessing
    union_list = list(set(to_predict_list).union(set(exist_list)))
    missing_elements = [elem for elem in required_elements if elem not in union_list]
    if missing_elements:
        print(f"Missing elements: {missing_elements} for postprocessing mode: {post_processing_mode}")
        print(f'Please fix json file and rerun')
        sys.exit(1)
    else:
        print("All required elements are present. Continuing...")
    

#   Validate if the input postprocessing mode is supported
def validate_postprocessing_mode(postprocessing_mode):
    supported_postprocessing_mode = ['none', 'generate_mask', 'fix_hole_overwrite', 'fix_hole_save_to_new']
    if postprocessing_mode not in supported_postprocessing_mode:
        print(f"Unsupported postprocessing mode: {postprocessing_mode}")
        print(f'Please choose from {supported_postprocessing_mode}')
        sys.exit(1)


#   To use for postprocessing
def postprocessing_packer(postprocessing_mode, OUTPUT_DIR, IQUV_DATA_DIR, use_parallel = False, max_parallel_workers = 4, skip_mask_generation = False):
    # skip_mask_generation: If True, will skip mask generation and assume that mask already exists
    
    validate_postprocessing_mode(postprocessing_mode)
    
    # Postprocessing
    if postprocessing_mode == 'none':
        print('No further postprocessing. Job completed.')
        return
    
    if skip_mask_generation == False:
        # First generate mask, which is required for all postprocessing modes
        print('Generating mask...')
        continuum_based_bad_point_identify(OUTPUT_DIR, IQUV_DATA_DIR, parallel=use_parallel, max_parallel_workers=max_parallel_workers)
    
    if postprocessing_mode == 'generate_mask':
        print('Mask generation complete. Job completed.')
        return
    
    print('Starting hole fixing...')
    
    if postprocessing_mode == 'fix_hole_overwrite':
        Interpolation_packer('Overwrite', OUTPUT_DIR, 'None', parallel=use_parallel, max_parallel_workers=max_parallel_workers)
    elif postprocessing_mode == 'fix_hole_save_to_new':
        Interpolation_packer('SaveNew', OUTPUT_DIR, os.path.join(OUTPUT_DIR, 'Br_hole_fixed', 'spDisambig_Br'), parallel=use_parallel, max_parallel_workers=max_parallel_workers)
    else:
        print(f"Unknown postprocessing mode: {postprocessing_mode}")
        sys.exit(1)
    print('Job Completed')
    

def SuperSynthIA_predict_packer(json_dir):
    #torch.use_deterministic_algorithms(True)
    
    # Default stuff
    BIN_FOLDER = "./BINS"
    Norm_params_FOLDER = "./Norm_params"
    supported_output_name = ['spDisambig_Bp', 'spDisambig_Br', 'spDisambig_Bt', 'spDisambig_Field_Azimuth_Disamb', 'spInv_aB', 
                 'spInv_Field_Azimuth', 'spInv_Field_Inclination', 'spInv_Stray_Light_Fill_Factor']
    
    json_data = parse_json_file(json_dir)
    
    # Accessing values using keys
    MODEL_LOCATION = json_data['MODEL_LOCATION']
    GLOBAL_DEVICE = json_data['GLOBAL_DEVICE']
    output_name_list = json_data['output_name_list']
    IQUV_DATA_DIR = json_data['IQUV_DATA_DIR']
    OUTPUT_DIR = json_data['OUTPUT_DIR']
    reproducible = json_data['reproducible']
    save_std = json_data['save_std']
    save_CI = json_data['save_CI']
    save_orig_logit = json_data['save_orig_logit']
    use_parallel = json_data['use_parallel']
    max_parallel_workers = json_data['max_parallel_workers']
    postprocessing_mode = json_data['postprocessing_mode']
    
    
    for output_name in output_name_list:
        if output_name not in supported_output_name:
            print(f"Unsupported output name: {output_name}")
            print(f'Please choose from {supported_output_name}')
            sys.exit(1)
    
    validate_postprocessing_mode(postprocessing_mode)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    
    # Save a copy of the json file
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json_path = os.path.join(OUTPUT_DIR, f'pred_{current_time}.json')
    with open(output_json_path, 'w') as output_file:
        json.dump(json_data, output_file, indent=4)
    
    
    # Verify if can do post_processing
    verify_have_necessary_preds(output_name_list, OUTPUT_DIR, postprocessing_mode)
    
    
    # For each output type, run the model in series
    for y_name in output_name_list:

        SAVE_PATH = os.path.join(OUTPUT_DIR, y_name)

        # Get the bins
        bin_name = y_name.replace("spDisambig_", "")
        bin_name = bin_name.replace("spInv_", "")
        bin_NPY_DIR = os.path.join(BIN_FOLDER, f'bins_{bin_name}.npy')
        bins = np.load(bin_NPY_DIR)
        bins = torch.from_numpy(bins)

        # Get the corresponding model assume that only one model for each folder
        C_MODEL_SAVE_PATH = os.path.join(MODEL_LOCATION, f'{y_name}')
        assert(len(os.listdir(C_MODEL_SAVE_PATH)) == 1)
        C_MODEL_SAVE_PATH = os.path.join(C_MODEL_SAVE_PATH, (os.listdir(C_MODEL_SAVE_PATH))[0])
        assert(y_name in C_MODEL_SAVE_PATH)


        # Need to pack these
        X_norm_DIR = os.path.join(Norm_params_FOLDER, "X_Norm_param.npy")
        y_norm_DIR = os.path.join(Norm_params_FOLDER, f"{y_name}_Norm_param.npy")



        print(f'Loading model from {C_MODEL_SAVE_PATH}')


        model = UNet(24 , 1, batchnorm=False, dropout=0.3, regression=False, bins=len(bins), bc=64).to(GLOBAL_DEVICE)



        model.load_state_dict(torch.load(C_MODEL_SAVE_PATH, map_location=GLOBAL_DEVICE)['model'])
        model.eval()

        print(f'Running on {GLOBAL_DEVICE}, producing {y_name} predictions')
        print(f'Saving to {SAVE_PATH}')

        inference_full_disk_80_days(IQUV_DATA_DIR, SAVE_PATH, model, X_norm_DIR, y_norm_DIR, GLOBAL_DEVICE, is_CLASSIFICATION=True,
                                    bins = bins, y_name = y_name, save_std = save_std, save_CI = save_CI, save_as_FITS = True,
                                    save_orig_logit = save_orig_logit, parallel=use_parallel, parallel_workers=max_parallel_workers, reproducible=reproducible)


    # Postprocessing
    if postprocessing_mode == 'none':
        print('No further postprocessing. Job completed.')
        return
    
    # First generate mask, which is required for all postprocessing modes
    print('Generating mask...')
    continuum_based_bad_point_identify(OUTPUT_DIR, IQUV_DATA_DIR, parallel=use_parallel, max_parallel_workers=max_parallel_workers)
    
    if postprocessing_mode == 'generate_mask':
        print('Mask generation complete. Job completed.')
        return
    
    print('Starting hole fixing...')
    
    if postprocessing_mode == 'fix_hole_overwrite':
        Interpolation_packer('Overwrite', OUTPUT_DIR, 'None', parallel=use_parallel, max_parallel_workers=max_parallel_workers)
    elif postprocessing_mode == 'fix_hole_save_to_new':
        Interpolation_packer('SaveNew', OUTPUT_DIR, os.path.join(OUTPUT_DIR, 'Br_hole_fixed', 'spDisambig_Br'), parallel=use_parallel, max_parallel_workers=max_parallel_workers)
    else:
        print(f"Unknown postprocessing mode: {postprocessing_mode}")
        sys.exit(1)
    print('Job Completed')
