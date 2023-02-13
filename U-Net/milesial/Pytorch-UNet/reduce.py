import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate
from unet import UNet
#from utils.data_loading import BasicDataset, CarvanaDataset
from utils.data_loading import CarvanaDataset
#from labml_nn.unet.carvana import CarvanaDataset
from utils.dice_score import dice_loss

dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')

dataset = CarvanaDataset(dir_img, dir_mask)
                        
print(type(dataset))

print(len(dataset))

