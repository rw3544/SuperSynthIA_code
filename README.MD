Thank you for your interest in SuperSynthIA! This file will guide you through the repo and explain how to use our code.
# Instructions:

### Environment Setup
This repo contains an environment.yml file that contains all necessary packages. Please run the following code to build a compatible conda environment.

    
```
conda env create -f environment.yml
```

    
### Inference:
1. Please check the [demo.json](./configs/demo.json) in ./configs folder as an example. Change the corresponding parameters and build your own json config file.
2. Go to [demo.py](./demo.py), change `config_location` to the location of your config file. Then, run `python demo.py`

For detailed explaination of each parameter in the config file, click here.

### Postprocessing:
SuperSynthIA can occasionally produce small holes in $B_r$ within large sunspots, particularly in regions where the inputs are similar to those in the quiet sun. To address this issue, we offer specialized postprocessing tools integrated in [postprocessing.py](./postprocessing.py):

1. Mask for pointing out pixels that the model is uncertain about in Br
2. Using thin-plate spline interpolation to fix holes detected (1), saving outputs to a separate directory
3. Same as (2), but overwrites predictions with corrected ones.



### FAQ
##### 1. CUDA out of memory
- 

### Some hard-coded stuff:
1. BINS/ for the bins to be used for classification for each output
2. Norm_params/ for the normalization parameters for each output

The reproducible results depend on timestamp from the fits file name, please note that

in continuum_process_file, IQUV_DIGIT needs to be fixed



### Config file Guide

1. **MODEL_LOCATION**
   - **Description**: Path to the directory containing the trained models.
   - **Example**: `"MODEL_LOCATION": "./Orig_Trained_Models"`

2. **GLOBAL_DEVICE**
   - **Description**: Device to be used for computation (e.g., "cuda" or "cpu").
   - **Example**: `"GLOBAL_DEVICE": "cuda"`

3. **output_name_list**
   - **Description**: List of output names to be generated. Choose from:
     - `"spDisambig_Bp"`
     - `"spDisambig_Br"`
     - `"spDisambig_Bt"`
     - `"spDisambig_Field_Azimuth_Disamb"`
     - `"spInv_aB"`
     - `"spInv_Field_Azimuth"`
     - `"spInv_Field_Inclination"`
   - **Example**: 
     ```json
     "output_name_list": [
         "spDisambig_Bp", 
         "spDisambig_Br", 
         "spDisambig_Bt", 
         "spInv_aB", 
         "spInv_Field_Inclination", 
         "spDisambig_Field_Azimuth_Disamb"
     ]
     ```

4. **IQUV_DATA_DIR**
   - **Description**: Directory containing the input data (IQUV fits files from hmi720s, can be downloaded from [JSOC](http://jsoc.stanford.edu/ajax/exportdata.html)).
   - **Example**: `"IQUV_DATA_DIR": "./SMALL_DATASET"`

5. **OUTPUT_DIR**
   - **Description**: Directory where the output will be saved.
   - **Example**: `"OUTPUT_DIR": "./Predictions"`

6. **reproducible**
   - **Description**: Whether to make the process reproducible using a fixed 64x64 triangular noise for dithering. This process guarentees when the environment and 'CHUNK_SIZE' in [full_disk_utils.py](./helpers/full_disk_utils.py) is fixed, the model will produce the same results. It also depends on timestamp extracted from input fits file name. Please note that this can slow things down by approximately 4x.
   - **Example**: `"reproducible": true`

7. **save_std**
   - **Description**: Whether to save the per-pixel standard deviation.
   - **Example**: `"save_std": false`

8. **save_CI**
   - **Description**: Whether to save the per-pixel confidence intervals (90%).
   - **Example**: `"save_CI": false"`

9. **save_orig_logit**
   - **Description**: Whether to save the original logits produced by the model, which can be used to calculate probabilities of each bin (about 5Gb per input).
   - **Example**: `"save_orig_logit": false"`

10. **use_parallel**
    - **Description**: Whether to use parallel processing.
    - **Example**: `"use_parallel": true"`

11. **max_parallel_workers**
    - **Description**: Maximum number of parallel workers. It is recommended to set this to the number of CPUs on your machine, and reduce if you encounter memory issues.
    - **Example**: `"max_parallel_workers": 8"`

12. **postprocessing_mode**
    - **Description**: Postprocessing mode. Choose from:
      - `"none"`
      - `"generate_mask"`
      - `"fix_hole_overwrite"`
      - `"fix_hole_save_to_new"`
    - **Note**: For producing aB/Inclination/Azimuth only, set to `"none"` to avoid any issues.
    - **Example**: `"postprocessing_mode": "none"`

### Example Configuration

Here is an example configuration file:

```json
{
    "MODEL_LOCATION": "./Orig_Trained_Models",

    "GLOBAL_DEVICE": "cuda",

    "output_name_list": ["spDisambig_Bp", "spDisambig_Br", "spDisambig_Bt", "spInv_aB", "spInv_Field_Inclination", "spDisambig_Field_Azimuth_Disamb"],

    "IQUV_DATA_DIR": "./SMALL_DATASET",

    "OUTPUT_DIR": "./Predictions",

    "reproducible": true,

    "save_std": false,

    "save_CI": false,

    "save_orig_logit": false,

    "use_parallel": true,

    "max_parallel_workers": 8,

    "postprocessing_mode": "none"
}