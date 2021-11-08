#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated script for the replication of results for AVIRIS images mblbcs2021.
V1.2
Sebastià Mijares i Verdú - GICI, UAB
sebastia.mijares@uab.cat

This script compiles the training and testing routines for models described in the paper "

Com utilitzar aquest codi
-------------------------

Executar acompanyat dels mòduls requerits i indicant els paràmetres desitjats a
les comandes. La comanda 'train' entrena un model de xarxa neural. La comanda
'test' el prova sobre totes les imatges d'un repositori. La geometria de les
imatges dels repositoris es carrega automàticament.

Requeriments
------------

-Mòdul argparse
-Mòdul glob
-Mòdul sys
-Mòdul absl
-Mòdul os
-Mòdul numpy
-Mòdul math

"""

import argparse
import sys
from absl import app
from absl.flags import argparse_flags
import os
import numpy as np
from math import log10, sqrt
import tensorflow as tf

def read_raw(filename,height,width,bands,endianess,datatype):
    """
    Function imported from metrics v2.1.
    """
    string = tf.io.read_file(filename)
    vector = tf.io.decode_raw(string,datatype,little_endian=(endianess==1))
    return tf.reshape(vector,[height,width,bands])

def ms_ssim(X, Y, height=512, width=680, bands=224, endianess = 1, data_type = tf.uint16, bit_length = 16):
    """
    Function imported from metrics v2.1.
    """
    x = read_raw(X,height,width,bands,endianess,data_type)
    x_hat = read_raw(Y,height,width,bands,endianess,data_type)
    return np.array(tf.squeeze(tf.image.ssim_multiscale(x, x_hat, 2**bit_length-1)))

def train(args):
    if not os.path.isdir('./models/'+args.model):
        os.mkdir('./models/'+args.model+'/')
        os.mkdir('./models/'+args.model+'/logs')
        os.mkdir('./models/'+args.model+'/interval_1')
        os.mkdir('./models/'+args.model+'/interval_2')
        os.mkdir('./models/'+args.model+'/interval_3')
        os.mkdir('./models/'+args.model+'/interval_4')
        os.mkdir('./models/'+args.model+'/interval_5')
    
    corpus = os.listdir('./datasets/'+args.dataset)
    #The images are sepparated into the collections of bands of each designated interval.
    os.mkdir('./datasets/'+args.dataset+'/interval_1')
    os.mkdir('./datasets/'+args.dataset+'/interval_2')
    os.mkdir('./datasets/'+args.dataset+'/interval_3')
    os.mkdir('./datasets/'+args.dataset+'/interval_4')
    os.mkdir('./datasets/'+args.dataset+'/interval_5')
    for IMAGE in corpus:
        if os.path.splitext(IMAGE)[1] == '.raw':
            org_arr = np.reshape(np.fromfile('./datasets/'+args.dataset+'/'+IMAGE,dtype=np.uint16),(224,args.width,args.height))
            name = IMAGE.split('.')[0]
            for i in range(224):
                new_arr = org_arr[i,:,:]
                if i <40:
                    new_arr.tofile('./datasets/interval_1/'+args.dataset+'/'+name+'_'+str(i)+'.1_'+str(args.width)+'_'+str(args.height)+'_2_1_0.raw')
                elif i<96:
                    new_arr.tofile('./datasets/interval_2/'+args.dataset+'/'+name+'_'+str(i)+'.1_'+str(args.width)+'_'+str(args.height)+'_2_1_0.raw')
                elif i<155:
                    new_arr.tofile('./datasets/interval_3/'+args.dataset+'/'+name+'_'+str(i)+'.1_'+str(args.width)+'_'+str(args.height)+'_2_1_0.raw')
                elif i <165:
                    new_arr.tofile('./datasets/interval_4/'+args.dataset+'/'+name+'_'+str(i)+'.1_'+str(args.width)+'_'+str(args.height)+'_2_1_0.raw')
                else:
                    new_arr.tofile('./datasets/interval_5/'+args.dataset+'/'+name+'_'+str(i)+'.1_'+str(args.width)+'_'+str(args.height)+'_2_1_0.raw')
    #Each model is trained sequentially.
    if args.hyperprior:
        os.system('python ./architectures/mblbcs2021_hyperprior --model_path ./models/'+args.model+' -V train --train_path ./models/'+args.model+'/logs --train_glob ".datasets/'+args.dataset+'/interval_1/*.raw" --num_scales '+str(args.num_scales)+' --scale_min '+str(args.scale_min)+' --scale_max '+str(args.scale_max)+' --epochs '+str(args.epochs)+' --bands '+str(224)+' --width '+str(args.width)+' --height '+str(args.height)+' --endianess 1 --lambda '+str(args.lmbda)+' --patchsize '+str(args.patchsize)+' --num_filters '+str(args.num_filters)+' --learning_rate '+str(args.learning_rate)+' --steps_per_epoch '+str(args.steps_per_epoch))
        os.system('python ./architectures/mblbcs2021_hyperprior --model_path ./models/'+args.model+' -V train --train_path ./models/'+args.model+'/logs --train_glob ".datasets/'+args.dataset+'/interval_2/*.raw" --num_scales '+str(args.num_scales)+' --scale_min '+str(args.scale_min)+' --scale_max '+str(args.scale_max)+' --epochs '+str(args.epochs)+' --bands '+str(224)+' --width '+str(args.width)+' --height '+str(args.height)+' --endianess 1 --lambda '+str(args.lmbda)+' --patchsize '+str(args.patchsize)+' --num_filters '+str(args.num_filters)+' --learning_rate '+str(args.learning_rate)+' --steps_per_epoch '+str(args.steps_per_epoch))
        os.system('python ./architectures/mblbcs2021_hyperprior --model_path ./models/'+args.model+' -V train --train_path ./models/'+args.model+'/logs --train_glob ".datasets/'+args.dataset+'/interval_3/*.raw" --num_scales '+str(args.num_scales)+' --scale_min '+str(args.scale_min)+' --scale_max '+str(args.scale_max)+' --epochs '+str(args.epochs)+' --bands '+str(224)+' --width '+str(args.width)+' --height '+str(args.height)+' --endianess 1 --lambda '+str(args.lmbda)+' --patchsize '+str(args.patchsize)+' --num_filters '+str(args.num_filters)+' --learning_rate '+str(args.learning_rate)+' --steps_per_epoch '+str(args.steps_per_epoch))
        os.system('python ./architectures/mblbcs2021_hyperprior --model_path ./models/'+args.model+' -V train --train_path ./models/'+args.model+'/logs --train_glob ".datasets/'+args.dataset+'/interval_4/*.raw" --num_scales '+str(args.num_scales)+' --scale_min '+str(args.scale_min)+' --scale_max '+str(args.scale_max)+' --epochs '+str(args.epochs)+' --bands '+str(224)+' --width '+str(args.width)+' --height '+str(args.height)+' --endianess 1 --lambda '+str(args.lmbda)+' --patchsize '+str(args.patchsize)+' --num_filters '+str(args.num_filters)+' --learning_rate '+str(args.learning_rate)+' --steps_per_epoch '+str(args.steps_per_epoch))
        os.system('python ./architectures/mblbcs2021_hyperprior --model_path ./models/'+args.model+' -V train --train_path ./models/'+args.model+'/logs --train_glob ".datasets/'+args.dataset+'/interval_5/*.raw" --num_scales '+str(args.num_scales)+' --scale_min '+str(args.scale_min)+' --scale_max '+str(args.scale_max)+' --epochs '+str(args.epochs)+' --bands '+str(224)+' --width '+str(args.width)+' --height '+str(args.height)+' --endianess 1 --lambda '+str(args.lmbda)+' --patchsize '+str(args.patchsize)+' --num_filters '+str(args.num_filters)+' --learning_rate '+str(args.learning_rate)+' --steps_per_epoch '+str(args.steps_per_epoch))
    else:
        os.system('python ./architectures/mblbcs2021_vanilla --model_path ./models/'+args.model+' -V train --train_path ./models/'+args.model+'/logs --train_glob ".datasets/'+args.dataset+'/interval_1/*.raw" --epochs '+str(args.epochs)+' --bands '+str(224)+' --width '+str(args.width)+' --height '+str(args.height)+' --endianess 1 --lambda '+str(args.lmbda)+' --patchsize '+str(args.patchsize)+' --num_filters '+str(args.num_filters)+' --learning_rate '+str(args.learning_rate)+' --steps_per_epoch '+str(args.steps_per_epoch))
        os.system('python ./architectures/mblbcs2021_vanilla --model_path ./models/'+args.model+' -V train --train_path ./models/'+args.model+'/logs --train_glob ".datasets/'+args.dataset+'/interval_2/*.raw" --epochs '+str(args.epochs)+' --bands '+str(224)+' --width '+str(args.width)+' --height '+str(args.height)+' --endianess 1 --lambda '+str(args.lmbda)+' --patchsize '+str(args.patchsize)+' --num_filters '+str(args.num_filters)+' --learning_rate '+str(args.learning_rate)+' --steps_per_epoch '+str(args.steps_per_epoch))
        os.system('python ./architectures/mblbcs2021_vanilla --model_path ./models/'+args.model+' -V train --train_path ./models/'+args.model+'/logs --train_glob ".datasets/'+args.dataset+'/interval_3/*.raw" --epochs '+str(args.epochs)+' --bands '+str(224)+' --width '+str(args.width)+' --height '+str(args.height)+' --endianess 1 --lambda '+str(args.lmbda)+' --patchsize '+str(args.patchsize)+' --num_filters '+str(args.num_filters)+' --learning_rate '+str(args.learning_rate)+' --steps_per_epoch '+str(args.steps_per_epoch))
        os.system('python ./architectures/mblbcs2021_vanilla --model_path ./models/'+args.model+' -V train --train_path ./models/'+args.model+'/logs --train_glob ".datasets/'+args.dataset+'/interval_4/*.raw" --epochs '+str(args.epochs)+' --bands '+str(224)+' --width '+str(args.width)+' --height '+str(args.height)+' --endianess 1 --lambda '+str(args.lmbda)+' --patchsize '+str(args.patchsize)+' --num_filters '+str(args.num_filters)+' --learning_rate '+str(args.learning_rate)+' --steps_per_epoch '+str(args.steps_per_epoch))
        os.system('python ./architectures/mblbcs2021_vanilla --model_path ./models/'+args.model+' -V train --train_path ./models/'+args.model+'/logs --train_glob ".datasets/'+args.dataset+'/interval_5/*.raw" --epochs '+str(args.epochs)+' --bands '+str(224)+' --width '+str(args.width)+' --height '+str(args.height)+' --endianess 1 --lambda '+str(args.lmbda)+' --patchsize '+str(args.patchsize)+' --num_filters '+str(args.num_filters)+' --learning_rate '+str(args.learning_rate)+' --steps_per_epoch '+str(args.steps_per_epoch))
    #The collections of individual bands generated are deleted.
    os.system('rm -r ./datasets/'+args.dataset+'/interval_1')
    os.system('rm -r ./datasets/'+args.dataset+'/interval_2')
    os.system('rm -r ./datasets/'+args.dataset+'/interval_3')
    os.system('rm -r ./datasets/'+args.dataset+'/interval_4')
    os.system('rm -r ./datasets/'+args.dataset+'/interval_5')
    
def test(args):
    corpus = os.listdir('.datasets/'+args.dataset)
    results = open('.models/'+args.model+'/test_results.csv','w')
    results.write('Image,Raw size,TFCI size,MSE,PSNR (dB)')
    if args.SSIM:
        results.write(',MS-SSIM')
    if args.SAM:
        results.write(',SAM (radians)')
    results.write ('\n')
    
    for IMAGE in corpus:
        if os.path.splitext(IMAGE)[1] == '.raw':
            path_to_image = './datasets/'+args.dataset+'/'+IMAGE
            original_size = os.stat(path_to_image)[6]
            results.write(IMAGE+','+str(original_size)+',')
            compressed_size = 0
            raw_tfci_path_to_image = path_to_image+'.tfci.raw'
            org_arr = np.reshape(np.fromfile('./datasets/'+args.dataset+'/'+IMAGE,dtype=np.uint16),(224,args.height,args.width))
            new_arr = []
            if args.hyperprior:
                for band in range(224):
                    path_to_band = './datasets/'+'/band'+str(band)+'.raw'
                    org_arr[band,:,:].tofile(path_to_band)
                    if band < 40:
                        os.system('python ./architectures/mblbcs2021_hyperprior --model_path ./models/'+args.model+'/interval_1 compress '+path_to_band+' 1 '+str(args.width)+' '+str(args.height)+' 1')
                        os.system('python ./architectures/mblbcs2021_hyperprior --model_path ./models/'+args.model+'/interval_1 decompress '+path_to_band+'.tfci 1 '+str(args.width)+' '+str(args.height)+' 1')
                    elif band < 96:
                        os.system('python ./architectures/mblbcs2021_hyperprior --model_path ./models/'+args.model+'/interval_2 compress '+path_to_band+' 1 '+str(args.width)+' '+str(args.height)+' 1')
                        os.system('python ./architectures/mblbcs2021_hyperprior --model_path ./models/'+args.model+'/interval_2 decompress '+path_to_band+'.tfci 1 '+str(args.width)+' '+str(args.height)+' 1')
                    elif band < 155:
                        os.system('python ./architectures/mblbcs2021_hyperprior --model_path ./models/'+args.model+'/interval_3 compress '+path_to_band+' 1 '+str(args.width)+' '+str(args.height)+' 1')
                        os.system('python ./architectures/mblbcs2021_hyperprior --model_path ./models/'+args.model+'/interval_3 decompress '+path_to_band+'.tfci 1 '+str(args.width)+' '+str(args.height)+' 1')
                    elif band < 165:
                        os.system('python ./architectures/mblbcs2021_hyperprior --model_path ./models/'+args.model+'/interval_4 compress '+path_to_band+' 1 '+str(args.width)+' '+str(args.height)+' 1')
                        os.system('python ./architectures/mblbcs2021_hyperprior --model_path ./models/'+args.model+'/interval_4 decompress '+path_to_band+'.tfci 1 '+str(args.width)+' '+str(args.height)+' 1')
                    else:
                        os.system('python ./architectures/mblbcs2021_hyperprior --model_path ./models/'+args.model+'/interval_5 compress '+path_to_band+' 1 '+str(args.width)+' '+str(args.height)+' 1')
                        os.system('python ./architectures/mblbcs2021_hyperprior --model_path ./models/'+args.model+'/interval_5 decompress '+path_to_band+'.tfci 1 '+str(args.width)+' '+str(args.height)+' 1')
                    new_band = np.reshape(np.fromfile(path_to_band+'.tfci.raw',dtype=np.uint16),(1,512,680)).astype(np.uint16)
                    compressed_size += os.stat(path_to_band+'.tfci')[6]
                    os.system('rm '+path_to_band)
                    os.system('rm '+path_to_band+'.tfci')
                    os.system('rm '+path_to_band+'.tfci.raw')
                    new_arr.append(new_band)
                new_arr = np.array(new_arr)
            else:
                for band in range(224):
                    path_to_band = './datasets/'+'/band'+str(band)+'.raw'
                    org_arr[band,:,:].tofile(path_to_band)
                    if band < 40:
                        os.system('python ./architectures/mblbcs2021_vanilla --model_path ./models/'+args.model+'/interval_1 compress '+path_to_band+' 1 '+str(args.width)+' '+str(args.height)+' 1')
                        os.system('python ./architectures/mblbcs2021_vanilla --model_path ./models/'+args.model+'/interval_1 decompress '+path_to_band+'.tfci 1 '+str(args.width)+' '+str(args.height)+' 1')
                    elif band < 96:
                        os.system('python ./architectures/mblbcs2021_vanilla --model_path ./models/'+args.model+'/interval_2 compress '+path_to_band+' 1 '+str(args.width)+' '+str(args.height)+' 1')
                        os.system('python ./architectures/mblbcs2021_vanilla --model_path ./models/'+args.model+'/interval_2 decompress '+path_to_band+'.tfci 1 '+str(args.width)+' '+str(args.height)+' 1')
                    elif band < 155:
                        os.system('python ./architectures/mblbcs2021_vanilla --model_path ./models/'+args.model+'/interval_3 compress '+path_to_band+' 1 '+str(args.width)+' '+str(args.height)+' 1')
                        os.system('python ./architectures/mblbcs2021_vanilla --model_path ./models/'+args.model+'/interval_3 decompress '+path_to_band+'.tfci 1 '+str(args.width)+' '+str(args.height)+' 1')
                    elif band < 165:
                        os.system('python ./architectures/mblbcs2021_vanilla --model_path ./models/'+args.model+'/interval_4 compress '+path_to_band+' 1 '+str(args.width)+' '+str(args.height)+' 1')
                        os.system('python ./architectures/mblbcs2021_vanilla --model_path ./models/'+args.model+'/interval_4 decompress '+path_to_band+'.tfci 1 '+str(args.width)+' '+str(args.height)+' 1')
                    else:
                        os.system('python ./architectures/mblbcs2021_vanilla --model_path ./models/'+args.model+'/interval_5 compress '+path_to_band+' 1 '+str(args.width)+' '+str(args.height)+' 1')
                        os.system('python ./architectures/mblbcs2021_vanilla --model_path ./models/'+args.model+'/interval_5 decompress '+path_to_band+'.tfci 1 '+str(args.width)+' '+str(args.height)+' 1')
                    new_band = np.reshape(np.fromfile(path_to_band+'.tfci.raw',dtype=np.uint16),(1,args.height,args.width)).astype(np.uint16)
                    compressed_size += os.stat(path_to_band+'.tfci')[6]
                    os.system('rm '+path_to_band)
                    os.system('rm '+path_to_band+'.tfci')
                    os.system('rm '+path_to_band+'.tfci.raw')
                    new_arr.append(new_band)
                new_arr = np.array(new_arr)
            if args.keep_reconstruction:
                new_arr.tofile(raw_tfci_path_to_image)
            #Currently this is only setup for 16-bit images; hard-coded.
            new_arr = new_arr.astype(np.float32)
            org_arr = org_arr.astype(np.float32)
            mse = np.mean((new_arr-org_arr)**2)
            if mse==0:
                psnr = 100
            else:
                psnr = 20*log10(65535/sqrt(mse))
            results.write(str(compressed_size)+','+str(mse)+','+str(psnr))
            if args.SSIM:
                ssim = ms_ssim(path_to_image, raw_tfci_path_to_image, height=args.height, bands=224, width=args.width, endianess=1)
                results.write(','+str(ssim))
            results.write('\n')
    results.close()

def parse_args(argv):
  """Parses command line arguments."""
  parser = argparse_flags.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # High-level options.
  parser.add_argument(
      "--model", default="test_model",
      help="Name of the model to be loaded. This must be the name of the directory generated by training, and stored in the ./models directory.")
  parser.add_argument(
      "--hyperprior", "-H", action="store_true",
      help="Select this option to train a hyperprior model. Otherwise, a model without a hyperprior will be trained.")
  subparsers = parser.add_subparsers(
      title="commands", dest="command",
      help="What to do: 'train' loads training data and trains (or continues "
           "to train) a new model. 'test' applies a trained model to all images "
           "in a repository and measures the rate-distortion performance.")

  # 'train' subcommand.
  train_cmd = subparsers.add_parser(
      "train",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Trains (or continues to train) a new model. Note that this "
                  "model trains on a continuous stream of patches drawn from "
                  "the training image dataset. An epoch is always defined as "
                  "the same number of batches given by --steps_per_epoch. "
                  "The purpose of validation is mostly to evaluate the "
                  "rate-distortion performance of the model using actual "
                  "quantization rather than the differentiable proxy loss. "
                  "Note that when using custom training images, the validation "
                  "set is simply a random sampling of patches from the "
                  "training set.")
  train_cmd.add_argument(
      "--lambda", type=float, default=0.01, dest="lmbda",
      help="Lambda for rate-distortion tradeoff.")
  train_cmd.add_argument(
      "--dataset", type=str, default="AVIRIS_uncal",
      help="Name of the training dataset. Will be searched in the ./datasets directory.")
  train_cmd.add_argument(
      "--num_filters", type=int, default=48,
      help="Number of filters per layer.")
  train_cmd.add_argument(
      "--patchsize", type=int, default=256,
      help="Size of image patches for training and validation.")
  train_cmd.add_argument(
      "--epochs", type=int, default=500,
      help="Train up to this number of epochs. (One epoch is here defined as "
           "the number of steps given by --steps_per_epoch, not iterations "
           "over the full training dataset.)")
  train_cmd.add_argument(
      "--steps_per_epoch", type=int, default=1000,
      help="Perform validation and produce logs after this many batches.")
  train_cmd.add_argument(
      "--learning_rate", type=float, default=0.0001, dest="learning_rate",
      help="Float indicating the learning rate for training.")
  train_cmd.add_argument(
      "--width", type=int, default=680, dest="width",
      help="Width of the images to train the model. All must be the same size.")
  train_cmd.add_argument(
      "--height", type=int, default=512, dest="height",
      help="Height of the images to train the model. All must be the same size.")
  train_cmd.add_argument(
      "--num_scales", type=int, default=64, dest="num_scales",
      help="Only for hyperprior models. Number of Gaussian scales to prepare range coding tables for.")
  train_cmd.add_argument(
      "--scale_min", type=float, default=0.11, dest="scale_min",
      help="Only for hyperprior models. Minimum value of standard deviation of Gaussians.")
  train_cmd.add_argument(
      "--scale_max", type=float, default=256.0, dest="scale_max",
      help="Only for hyperprior models. Maximum value of standard deviation of Gaussians.")

    # 'test' subcommand.
  test_cmd = subparsers.add_parser(
      "test",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Tests a trained model on all images in a dataset."
                  "the dataset's specifications will be loaded automatically")

  # Arguments for test command.
  test_cmd.add_argument(
      "--dataset", type=str, default="AVIRIS_uncal",
      help="Test dataset. Its geometry will be automatically loaded.")
  test_cmd.add_argument(
      "--SSIM", action="store_true",
      help="Computes MS-SSIM distortion.")
  test_cmd.add_argument(
      "--width", type=int, default=680, dest="width",
      help="Width of the images to train the model. All must be the same size.")
  test_cmd.add_argument(
      "--height", type=int, default=512, dest="height",
      help="Height of the images to train the model. All must be the same size.")
  test_cmd.add_argument(
      "--keep_reconstruction",  action="store_true", dest="keep_reconstruction",
      help="Keep the decompressed images for assessment.")

  # Parse arguments.
  args = parser.parse_args(argv[1:])
  if args.command is None:
    parser.print_usage()
    sys.exit(2)
  return args


def main(args):
  # Invoke subcommand.
  if args.command == "train":
    train(args)
  elif args.command == "test":
    test(args)

if __name__ == "__main__":
  app.run(main, flags_parser=parse_args)
