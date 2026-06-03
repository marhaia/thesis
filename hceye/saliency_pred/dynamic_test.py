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
import numpy as np, cv2
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from scipy.stats import multivariate_normal
from dataloader import DynamicDataset
from loss import *
from tqdm import tqdm
from utils import *
import random

parser = argparse.ArgumentParser()

parser.add_argument('--model_val_path',default="saved_models/pair.pt", type=str)
parser.add_argument('--no_workers',default=4, type=int)
parser.add_argument('--enc_model',default="pnas", type=str)
parser.add_argument('--validate',default=1, type=int)
parser.add_argument('--save_results',default=1, type=int)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from model import PNASModel, HighlightModel
# model = HighlightModel(num_channels=6, train_enc=True, load_weight=0)
# model = PNASModel()
model =  HighlightModel()
model = nn.DataParallel(model)
model.load_state_dict(torch.load(args.model_val_path))

model = model.to(device)



def validate(model, loader, device, args):
    model.eval()
    tic = time.time()
    cc_loss = AverageMeter()
    kldiv_loss = AverageMeter()
    nss_loss = AverageMeter()
    sim_loss = AverageMeter()
    auc_loss = AverageMeter()

    # for (img_1, gt, fixations) in loader:
    for (img_1, img_2), gt, fixations in loader:
        img_1 = img_1.to(device)
        img_2 = img_2.to(device)
        gt = gt.to(device)
        fixations = fixations.to(device)

        # pred_map = model(img_1)
        pred_map = model(img_1,img_2)

        cc_loss.update(cc(pred_map , gt))
        kldiv_loss.update(kldiv(pred_map , gt))
        nss_loss.update(nss(pred_map , fixations))
        sim_loss.update(similarity(pred_map , gt))

        # Compute AUC Judd score
        auc_score = auc_judd(pred_map, fixations, jitter=True, toPlot=False, normalize=False)
        if not np.isnan(auc_score):
            auc_loss.update(auc_score)

    print('AUC:{:.5f}, CC: {:.5f}, KLDIV: {:.5f}, NSS: {:.5f}, SIM: {:.5f}, Time:{:3f} minutes'.format(
        auc_loss.avg, cc_loss.avg, kldiv_loss.avg, nss_loss.avg, sim_loss.avg, (time.time() - tic) / 60))
    sys.stdout.flush()

    return cc_loss.avg

def main():
    if args.validate:
        print('validate')
        # Ensure the base directory ends with a slash
        base_dir = r"C:\Users\wuzek\OneDrive\Desktop\NewProject\highlight_data\dynamic"

        test_img_dir = os.path.join(base_dir, "images/val/")
        test_gt_dir = os.path.join(base_dir, "maps/val/")
        test_fix_dir = os.path.join(base_dir, "fixations/val/")


        val_dataset =DynamicDataset(img_dir=test_img_dir  , gt_dir=test_gt_dir, fix_dir=test_fix_dir, mode='pair')

        # Create a DataLoader instance
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


        with torch.no_grad():
            validate(model, val_loader, device, args)

if __name__ == '__main__':
    main()