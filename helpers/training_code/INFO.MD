This folder contains the training code for SuperSynthIA, provided for those who are interested. Our training dataset (~1TB) will also be released soon. Please note that this code is not actively maintained, as we are currently developing an improved version that we hope to release in the near future.

# Training Steps

## Preparing the Dataset:
1. Build zarr files based on the year from co-aligned scans (Relevant code: [data_preprocessing.py](./data_preprocessing.py); [build_pool_zarr.py](./build_pool_zarr.py)).
2. Choose a train/validation/test split, combine the zarr files, and build the respective datasets (Relevant code: [build_train_data.py](./build_train_data.py)).

## Training:
3. Use [train_pipeline.py](./train_pipeline.py) as an example to run the training code. Refer to [train.py](./train.py) for supporting functions.