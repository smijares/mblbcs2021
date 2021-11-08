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
Download our test dataset from [this Google Drive repository](https://drive.google.com/drive/folders/1GZarLjBJ7oBzm6D0ZLOtGnGUQTfZ-aDc?usp=sharing) and store it in a directory at `./datasets` (recommended name `AVIRIS_uncal`). You will find our test data is compressed as `.bzip2` files. To decompress them, use:

```
bunzip2 ./datasets/AVIRIS_uncal_crop/*
```

from the root of our directories.

The `main.py` script is designed to fetch datasets ad directories at `./datasets`, such as `./datasets/your_dataset`.

This code is hard-coded to **only read images with 224 bands** in `.raw` files stored as 16-bit unsigned integers in little-endian byte order, ordered in BSQ, splitting them into the 5 spectral intervals described in the paper. Make sure you process your data to match these specifications.

### Testing a trained model
To test a model (or collection of 5 of them) on a testing dataset use the `test` function of the `main.py` script. For general help on the commands, use:
```
python main.py -h
python main.py test -h
```
For example, if you have downloaded and decompressed our test data following our recommendations, the following command:
```
python main.py --model hyperprior_qual1 -H test --dataset AVIRIS_uncal --height 512 --width 680 --SSIM
```
Will test the hyperprior (`-H` option) model at quality 1 (`--model hyperprior_qual1`) on the `./datasets/AVIRIS_uncal` dataset, whose images all have height 512 and width 680, and record PSNR and MS-SSIM (`--SSIM`option) in the results file. This results file will be a `.csv` table stored in the model's directory.

The different pre-trained models available are in the `./models` directory. Their naming convention indicates the architecture used and a quality parameter (the higher, the more quality reconstruction).

### Training your own model
To train a model (or collection of 5 of them) on a given dataset use the `train` function of the `main.py` script. For general help on the commands, use:
```
python main.py -h
python main.py train -h
```
For example, if you have downloaded and decompressed our test data following our recommendations, use the following command:
```
python main.py --model new_vanilla_model train --dataset AVIRIS_uncal --height 512 --width 680 --epochs 500 --steps_per_epoch 1000
```

### Compressing and decompressing individual images
