import argparse
import glob, os
import torch
import sys
import time
import torch.nn as nn
import pickle
from torch.distributions.multivariate_normal import MultivariateNormal as Norm
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
from scipy.stats import multivariate_normal
from dataloader import SaliconDataset, HighlightDataset
from loss import *
import cv2
from utils import blur, AverageMeter
from torchsummary import summary
from model import VGGModel
from fvcore.nn import FlopCountAnalysis

parser = argparse.ArgumentParser()
parser.add_argument('--no_epochs',default=40, type=int)
parser.add_argument('--lr',default=1e-4, type=float)
parser.add_argument('--kldiv',default=True, type=bool)
parser.add_argument('--cc',default=False, type=bool)
parser.add_argument('--nss',default=False, type=bool)
parser.add_argument('--sim',default=False, type=bool)
parser.add_argument('--nss_emlnet',default=False, type=bool)
parser.add_argument('--nss_norm',default=False, type=bool)
parser.add_argument('--l1',default=False, type=bool)
parser.add_argument('--lr_sched',default=True, type=bool)
parser.add_argument('--dilation',default=False, type=bool)
parser.add_argument('--enc_model',default="pnas", type=str)
parser.add_argument('--optim',default="Adam", type=str)

parser.add_argument('--load_weight',default=0, type=int)
parser.add_argument('--kldiv_coeff',default=1.0, type=float)
parser.add_argument('--step_size',default=5, type=int)
parser.add_argument('--cc_coeff',default=-1.0, type=float)
parser.add_argument('--sim_coeff',default=-1.0, type=float)
parser.add_argument('--nss_coeff',default=1.0, type=float)
parser.add_argument('--nss_emlnet_coeff',default=1.0, type=float)
parser.add_argument('--nss_norm_coeff',default=1.0, type=float)
parser.add_argument('--l1_coeff',default=1.0, type=float)
parser.add_argument('--train_enc',default=0, type=int)

parser.add_argument('--dataset_dir',default="../highlight_data/", type=str)
parser.add_argument('--batch_size',default=32, type=int)
parser.add_argument('--log_interval',default=60, type=int)
parser.add_argument('--no_workers',default=4, type=int)
parser.add_argument('--model_val_path',default="../saved_models/salicon_pnas.pt", type=str)
parser.add_argument('--model_saved_path',default="../saved_models/hl_abs.pt", type=str)

args = parser.parse_args()

# Instantiate the model
model = VGGModel(train_enc=bool(args.train_enc), load_weight=args.load_weight)

# Check if CUDA is available and move the model to GPU if it is
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Apply DataParallel after moving the model to GPU
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

# Count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Number of trainable parameters:", count_parameters(model))

from ptflops import get_model_complexity_info

flops, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True, print_per_layer_stat=True)
print(f"FLOPs: {flops}")
print(f"Params: {params}")