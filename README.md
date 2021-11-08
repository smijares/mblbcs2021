# Hyperspectral remote sensing data compressionwith neural networks (2021) public code
Public code repository with the code for the paper "Hyperspectral remote sensing data compressionwith neural networks", by S. Mijares, J. Ballé, V. Laparra, J. Bartrina, M. Hernández-Carbonero, and J. Serra-Sagristà. Download this repository to test and replicate our results described in the paper.

## Description
This repository contains scripts to train and run networks such as those published in the "Hyperspectral remote sensing data compressionwith neural networks" paper, by S. Mijares, J. Ballé, V. Laparra, J. Bartrina, M. Hernández-Carbonero, and J. Serra-Sagristà. This includes the scripts for each of the architectures (generating new networks or running existing ones), the models we trained for the results published, and a core script that integrates the uses of our code. More on how to use this code in the "How to use this code" subsection below.

The test images for which we published results can be found in this [Google Drive repository](https://drive.google.com/drive/folders/1GZarLjBJ7oBzm6D0ZLOtGnGUQTfZ-aDc?usp=sharing). They are processed in the format compatible with this code: `.raw` files stored as 16-bit unsigned integers in little-endian byte order, ordered in BSQ. More on the data compatibility in the "How to use this code" subsection below.

## How to use this code
Download the repository *as is* and follow the indications below for how to replicate our published results, test our models on other datasets, or train new models from scratch.

### Installation
Download this repository *as is* in the environment you wish to run our code. Make sure you have the latest versions of `tensorflow` and `tensorflow-compression` installed using:

```
pip install tensorflow
pip install tensorflow-compression
```

### Test data
Download our test dataset from [this Google Drive repository](https://drive.google.com/drive/folders/1GZarLjBJ7oBzm6D0ZLOtGnGUQTfZ-aDc?usp=sharing) and store it in a directory at `./datasets` (recommended name `AVIRIS_uncal_crop`). You will find our test data is compressed as `.bzip2` files. To decompress them, use:

```
bunzip2 ./datasets/AVIRIS_uncal_crop/*
```

from the root of our directories.

The `main.py` script is designed to fetch datasets ad directories at `./datasets`, such as `./datasets/your_dataset`.

This code is hard-coded to **only read images with 224 bands** in `.raw` files stored as 16-bit unsigned integers in little-endian byte order, ordered in BSQ, splitting them into the 5 spectral intervals described in the paper. Make sure you process your data to match these specifications.

### Testing a trained model


### Training your own model
