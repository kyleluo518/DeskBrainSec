import os
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import pyvips as Vips
from tqdm import tqdm
from utils import vips_utils, normalize
from torchvision import transforms, utils
import time
import torchvision.models as models
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image, ImageFile
import statistics
from typing import Optional, Tuple
import pylibczi
from pylibczi import CziScene
import czifile
from czifile import CziFile 
import xml.etree.ElementTree as ET
import argparse
import gc 
import psutil
import resource
import platform
import pickle
import xmltodict
import time
import matplotlib.path as mpath
from skimage.draw import polygon
import cv2
import copy
import shutil

import torch
import random
import torchvision
import torch.nn as nn
from skimage import io, transform
from skimage.morphology import convex_hull_image

import os
import math

import scipy.ndimage
import skimage.io
from skimage.measure import label, regionprops
from skimage.morphology import square
from skimage.color import rgb2gray
from skimage import img_as_ubyte
from sklearn.metrics import confusion_matrix
from skimage.transform import resize

from bs4 import BeautifulSoup

import argparse

NP_DTYPE_TO_VIPS_FORMAT = {
        np.dtype('int8'): Vips.BandFormat.CHAR,
        np.dtype('uint8'): Vips.BandFormat.UCHAR,
        np.dtype('int16'): Vips.BandFormat.SHORT,
        np.dtype('uint16'): Vips.BandFormat.USHORT,
        np.dtype('int32'): Vips.BandFormat.INT,
        np.dtype('float32'): Vips.BandFormat.FLOAT,
        np.dtype('float64'): Vips.BandFormat.DOUBLE
    }

VIPS_FORMAT_TO_NP_DTYPE = {v:k for k, v in NP_DTYPE_TO_VIPS_FORMAT.items()}


def array_vips(vips_image, verbose=False):
    # dtype = np.dtype('u{}'.format(vips_image.BandFmt.bit_length() + 1))
    dtype = VIPS_FORMAT_TO_NP_DTYPE[vips_image.format]
    if verbose:
        print(dtype, vips_image.height, vips_image.width, vips_image.bands)
    return (np.fromstring(vips_image.write_to_memory(), dtype=dtype) #np.uint8)
             .reshape(vips_image.height, vips_image.width, vips_image.bands)).squeeze()


#Function using PyVips to tile the WSI
def save_and_tile(image_to_segment, imagename, output_dir, tile_size=3072):
    base_dir_name = os.path.join(output_dir, imagename)
    print(base_dir_name)
    if not os.path.exists(base_dir_name):
        os.makedirs(base_dir_name)
    Vips.Image.dzsave(image_to_segment, base_dir_name,
                        layout='google',
                        suffix='.jpg[Q=90]',
                        tile_size=tile_size,
                        depth='one',
                        properties=True)
    return None


def grabCZI(path, verbose = False):
    img = czifile.imread(path)
    if verbose:
        print(img.shape)
        print(img)
    
    img = np.array(img, dtype = np.uint8)
    
    scenes = img.shape[0]
    time = img.shape[1]
    height = img.shape[2]
    width = img.shape[3]
    channels = img.shape[4]
    
    
    img = img.reshape((height, width, channels))
    if verbose:
        print(img)
        print(img.shape) 
        
    dtype_to_format = {
        'uint8': 'uchar',
        'int8': 'char',
        'uint16': 'ushort',
        'int16': 'short',
        'uint32': 'uint',
        'int32': 'int',
        'float32': 'float',
        'float64': 'double',
        'complex64': 'complex',
        'complex128': 'dpcomplex',
    }
    
    ###codes from numpy2vips
    height, width, bands = img.shape
    img = img.reshape(width * height * bands)
    vips = Vips.Image.new_from_memory(img.data, width, height, bands,
                                      dtype_to_format['uint8'])
    
    return vips


WSI_DIR = ''  #TO-DO: add path to your folder of WSIs
SAVE_DIR = '' #TO-DO: add path to your saved tiles
# source: BrainSec svs_to_png.py
TILE_SIZE = 30000

wsi_slides = os.listdir(WSI_DIR)
imagenames = sorted(wsi_slides)
print("All WSIs in wsi_dir: ")
print(imagenames)

if not os.path.exists(SAVE_DIR):
    print("Tile folder you provided us does not exist, being created now...")
    os.makedirs(SAVE_DIR)



print("Starting tiling....")
for imagename in tqdm(imagenames[:]):
    start = time.time()
    if imagename.split('.')[-1] == 'svs':
        NAID = imagename.split('.')[0]
        print("Loading", imagename, " ......")
        vips_img = Vips.Image.new_from_file(WSI_DIR + imagename, level=0)
            
        print("Loaded Image: " + WSI_DIR + imagename)    
        
        save_and_tile(vips_img, os.path.splitext(imagename)[0], SAVE_DIR, tile_size = TILE_SIZE)
        print("Done Tiling: ", WSI_DIR + imagename)
        
    elif imagename.split('.')[-1] == 'czi':
        NAID = imagename.split('.')[0]
        print("Loading", imagename, " ......")

        vips_img = grabCZI(WSI_DIR + imagename)
        print("Loaded Image: " + WSI_DIR + imagename)
    
        save_and_tile(vips_img, os.path.splitext(imagename)[0], SAVE_DIR, tile_size = TILE_SIZE)

        print("Done Tiling: ", WSI_DIR + imagename)
              

    else:
        print("Skipped,", imagename, '. This file is either not .czi or .svs, or not the file assigned')
    
    
    print("processed in ", time.time()-start," seconds")
    print("____________________________________________")