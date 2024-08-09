Thank you for your interest in [SuperSynthIA](https://rw3544.github.io/SuperSynthIA/)! You can find the project website [here](https://rw3544.github.io/SuperSynthIA/). (https://rw3544.github.io/SuperSynthIA/) This file will guide you through the repo and explain how to use our code. <br>
**If you want us to produce data for you, please fill out this [form](https://forms.gle/8XyaXGc1q9wVBULi8).**
# Instructions:

## Environment Setup
This repo contains an environment.yml file that contains all necessary packages. Please run the following code to build a compatible conda environment.

    
```
conda env create -f environment.yml
conda activate SYIA
```

## Download model weights
* The trained model weights can be downloaded [here](https://www.dropbox.com/scl/fi/0efn6xx8cwupyeejgtg7l/SYIA_ORIG_MODEL.tar.gz?rlkey=xfv7xu9ep4hbzz2zcxzzcvxb4&st=opew50xc&dl=0). (https://www.dropbox.com/scl/fi/0efn6xx8cwupyeejgtg7l/SYIA_ORIG_MODEL.tar.gz?rlkey=xfv7xu9ep4hbzz2zcxzzcvxb4&st=opew50xc&dl=0) <br>
In our default setup, it is recommended to go to the root folder of this repository and then run 
```
wget -O SYIA_ORIG_MODEL.tar.gz "https://www.dropbox.com/scl/fi/0efn6xx8cwupyeejgtg7l/SYIA_ORIG_MODEL.tar.gz?rlkey=xfv7xu9ep4hbzz2zcxzzcvxb4&st=2pu3yzpw&dl=1"
tar -xzf SYIA_ORIG_MODEL.tar.gz
```
Then, the model can be found in [./Orig_Trained_Models](./Orig_Trained_Models)

* SuperSynthIA runs on the hmi.M_720s data series as inputs. One can download such data from [JSOC](http://jsoc.stanford.edu/ajax/exportdata.html).
We also provide a small dataset for your convenience to test the code:

```
wget -O SYIA_SMALL_DATASET.tar.gz "https://www.dropbox.com/scl/fi/6fivy89cq8f5xcdrrf7ug/SYIA_SMALL_DATASET.tar.gz?rlkey=erj4gth5rru85eamiqr2aunnq&st=y8sw3uu7&dl=1"
tar -xzf SYIA_SMALL_DATASET.tar.gz 
```
The dataset can then be found in [./SMALL_DATASET](./SMALL_DATASET)
    
## Inference:
After you have set up conda environment and downloaded model weights:
1. Please check the [demo.json](./configs/demo.json) in ./configs folder as an example. Change the corresponding parameters and build your own json config file.
2. Go to [demo.py](./demo.py), change `config_location` to the location of your config file. Then, run `python demo.py`

For detailed explaination of each parameter in the config file, click [here](#config-file-guide).

## Postprocessing:
### Holes in $B_r$

SuperSynthIA can occasionally produce small holes in $B_r$ within large sunspots, particularly in regions where the inputs are similar to those in the quiet sun. To address this issue, we offer specialized postprocessing tools integrated in [postprocessing.py](./postprocessing.py):

1. Mask for pointing out pixels that the model is uncertain about in Br
2. Using thin-plate spline interpolation to fix holes detected (1), saving outputs to a separate directory
3. Same as (2), but overwrites predictions with corrected ones.

### Analytic Method for producing $B_r$ , $B_p$ , $B_t$

SuperSynthIA supports analytic method to produce disambiguated components ($\alpha B_r$, $\alpha B_p$, $\alpha B_t$) from $\alpha B$, inclination, and azimuth. An example can be found in [analytic_disambig.py](./analytic_disambig.py).


## FAQ
### 1. CUDA out of memory
- This issue is typically caused by the `CHUNK_SIZE` parameter in [full_disk_utils.py](./helpers/full_disk_utils.py) being set too high. For an A40 GPU with 48GB of memory, we default this value to 1024 to optimize processing speed. However, for GPUs with smaller memory capacities, it is recommended to use a `CHUNK_SIZE` of 256. Please ensure that `CHUNK_SIZE` is a multiple of 64 and that 4096 is divisible by `CHUNK_SIZE`.

### 2. Some hard-coded stuff
1. BINS/ for the bins to be used for classification for each output
2. Norm_params/ for the normalization parameters for each output


### 3. What does reproducible means?

We employ logit dithering to introduce noise into the log-probabilities predicted by the model, which can cause slight variations in predictions for the same input. To ensure reproducibility, we use the timestamp and patch location as keys to calculate a seed for generating deterministic noise. 


### 4. How is the mask in [postprocessing](#holes-in) calculated?

Given the SuperSynthIA predictions for $B_r$, $B_p$, and $B_t$, we can construct a proxy for the continuum intensity and compare it with the actual continuum intensity. Significant differences between the two indicate that the SuperSynthIA predictions may not be reliable. If you are interested in the details, check `continuum_process_file` in [full_disk_utils.py](./helpers/full_disk_utils.py).



## Config file Guide

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
   - **Description**: Whether to make the process reproducible using a fixed 64x64 triangular noise for dithering. This process guarentees when the environment and 'CHUNK_SIZE' in [full_disk_utils.py](./helpers/full_disk_utils.py) is fixed, the model will produce the same results. It also depends on timestamp extracted from input fits file name. For a more detailed explaination, check [here](#3-what-does-reproducible-means).
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

## Example Configuration

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