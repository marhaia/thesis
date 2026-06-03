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
from dataloader import DynamicDataset
from loss import *
import cv2
from utils import blur, AverageMeter
from torchsummary import summary
import random
from model import PNASModel, HighlightModel

parser = argparse.ArgumentParser()
parser.add_argument('--no_epochs',default=200, type=int)
parser.add_argument('--lr',default=1e-6, type=float)
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
parser.add_argument('--kldiv_coeff',default=10.0, type=float)
parser.add_argument('--step_size',default=0, type=int)
parser.add_argument('--cc_coeff',default=-5.0, type=float)
parser.add_argument('--sim_coeff',default=0.0, type=float)
parser.add_argument('--nss_coeff',default=-1.0, type=float)
parser.add_argument('--nss_emlnet_coeff',default=1.0, type=float)
parser.add_argument('--nss_norm_coeff',default=1.0, type=float)
parser.add_argument('--l1_coeff',default=1.0, type=float)
parser.add_argument('--train_enc',default=0, type=int)

parser.add_argument('--dataset_dir',default="../highlight_data/", type=str)
parser.add_argument('--batch_size',default=32, type=int)
parser.add_argument('--log_interval',default=60, type=int)
parser.add_argument('--no_workers',default=4, type=int)
parser.add_argument('--model_saved_path',default="saved_models/pair.pt", type=str)
parser.add_argument('--model_pretrained_path',default="saved_models/highlight.pt", type=str)

args = parser.parse_args()


pretrained_weights = torch.load(args.model_pretrained_path)


# model = PNASModel()

# model.load_state_dict(torch.load(args.model_pretrained_path))
model = HighlightModel(train_enc=bool(args.train_enc), load_weight=args.load_weight)

model = nn.DataParallel(model)
# Prepare the new model's state_dict for weight transfer
model_state_dict = model.state_dict()
# Transfer weights, skipping the channel_reduction layer
for name, param in pretrained_weights.items():
    if name in model_state_dict and 'channel_reduction' not in name:
        # Here we make sure the layer exists in the new model and isn't the channel reduction layer
        model_state_dict[name].copy_(param)


model.load_state_dict(model_state_dict)

print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Ensure the base directory ends with a slash
base_dir = r"C:\Users\wuzek\OneDrive\Desktop\NewProject\highlight_data\dynamic"

train_img_dir = os.path.join(base_dir, "images/train/")
train_gt_dir = os.path.join(base_dir, "maps/train/")
train_fix_dir = os.path.join(base_dir, "fixations/train/")

val_img_dir = os.path.join(base_dir, "images/val/")
val_gt_dir = os.path.join(base_dir, "maps/val/")
val_fix_dir = os.path.join(base_dir, "fixations/val/")

# Create the dataset for highlighted images
train_dataset = DynamicDataset(img_dir=train_img_dir , gt_dir=train_gt_dir, fix_dir=train_fix_dir, mode='pair')
val_dataset = DynamicDataset(img_dir=val_img_dir , gt_dir=val_gt_dir, fix_dir=val_fix_dir, mode='pair')

# Create a DataLoader instance
train_loader = DataLoader(train_dataset , batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset , batch_size=1, shuffle=True)
test_loader = DataLoader(val_dataset , batch_size=1, shuffle=False)

def loss_func(pred_map, gt, fixations, args):
    criterion = nn.L1Loss()  # This seems unused, consider removing if not needed later

    # Calculate each loss component separately and then add them together
    loss_kldiv = args.kldiv_coeff * kldiv(pred_map, gt)
    loss_cc = args.cc_coeff * cc(pred_map, gt)
    loss_nss = args.nss_coeff * nss(pred_map, fixations)
    loss_sim = args.sim_coeff * similarity(pred_map, gt)

    # Sum all the loss components into a single tensor
    total_loss = loss_kldiv + loss_cc + loss_nss + loss_sim
    return total_loss


def train(model, optimizer, loader, epoch, device, args):
    model.train()
    tic = time.time()

    total_loss = 0.0
    cur_loss = 0.0

    # for idx, (img_1,  gt, fixations) in enumerate(loader):
    #     img_1 = img_1.to(device)
    for batch_idx, ((img_1, img_2), gt, fixations) in enumerate(train_loader):
        img_1, img_2, gt, fixations = img_1.to(device), img_2.to(device), gt.to(device), fixations.to(device)

        gt = gt.to(device)
        fixations = fixations.to(device)

        optimizer.zero_grad()
        # pred_map = model(img_1)
        pred_map = model(img_1, img_2)
        assert pred_map.size() == gt.size()
        loss = loss_func(pred_map, gt, fixations, args)
        loss.backward()
        total_loss += loss.item()
        cur_loss += loss.item()

        optimizer.step()
        # if idx % args.log_interval == (args.log_interval - 1):
        #     print(
        #         '[{:2d}, {:5d}] avg_loss : {:.5f}, time:{:3f} minutes'.format(epoch, idx, cur_loss / args.log_interval,
        #                                                                       (time.time() - tic) / 60))
        #     cur_loss = 0.0
        #     sys.stdout.flush()

    print('[{:2d}, train] avg_loss : {:.5f}'.format(epoch, total_loss / len(loader)))
    sys.stdout.flush()

    return total_loss / len(loader)


def validate(model, loader, epoch, device):
    model.eval()
    tic = time.time()
    # Assuming AverageMeter and other necessary imports are defined elsewhere
    cc_loss = AverageMeter()
    kldiv_loss = AverageMeter()
    nss_loss = AverageMeter()
    sim_loss = AverageMeter()

    for ((img_1, img_2), gt, fixations) in loader:
    # for (img_1, gt, fixations) in loader:
        img_1, img_2, gt, fixations = img_1.to(device), img_2.to(device), gt.to(device), fixations.to(device)
        # img_1,  gt, fixations = img_1.to(device),  gt.to(device), fixations.to(device)

        with torch.no_grad():
            pred_map = model(img_1, img_2)
            # pred_map = model(img_1)

        # # Ensure pred_map is detached and properly sized
        # pred_map = pred_map.detach().cpu()


        cc_loss.update(cc(pred_map , gt))
        kldiv_loss.update(kldiv(pred_map , gt))
        nss_loss.update(nss(pred_map , fixations))
        sim_loss.update(similarity(pred_map , gt))

    print('[{:2d},   val] CC : {:.5f}, KLDIV : {:.5f}, NSS : {:.5f}, SIM : {:.5f}  time:{:3f} minutes'.format(epoch,
                                                                                                              cc_loss.avg,
                                                                                                              kldiv_loss.avg,
                                                                                                              nss_loss.avg,
                                                                                                              sim_loss.avg,
                                                                                                              (
                                                                                                                          time.time() - tic) / 60))
    sys.stdout.flush()


# Initialize the optimizer and scheduler
params = list(filter(lambda p: p.requires_grad, model.parameters()))
optimizer = torch.optim.Adam(params, lr=args.lr)  # Example using Adam
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

# Initialize early stopping parameters
patience = 10
counter = 0
best_loss = float('inf')
early_stop = False

for epoch in range(args.no_epochs):
    torch.autograd.set_detect_anomaly(True)
    print(f"Starting epoch {epoch + 1}/{args.no_epochs} ...")

    # Train
    print(f"Training epoch {epoch + 1} ...")
    model.train()
    validate(model, test_loader, epoch, device)
    total_loss = 0.0
    # for img_1, gt, fixations in train_loader:
    for (img_1,img_2), gt, fixations in train_loader:
        # img_1, gt, fixations = img_1.to(device), gt.to(device), fixations.to(device)
        img_1, img_2, gt, fixations = img_1.to(device), img_2.to(device), gt.to(device), fixations.to(device)


        optimizer.zero_grad()
        pred_map = model(img_1,img_2)
        # pred_map = model(img_1)
        assert pred_map.size() == gt.size()
        loss = loss_func(pred_map, gt, fixations, args)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1} - Training loss: {avg_train_loss:.4f}")

    # Validationen
    print(f"Validating epoch {epoch + 1} ...")
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        # for (img_1, gt, fixations) in val_loader:
        for ((img_1, img_2), gt, fixations) in val_loader:
            img_1, img_2, gt, fixations = img_1.to(device), img_2.to(device), gt.to(device), fixations.to(device)
            # img_1,gt, fixations = img_1.to(device),gt.to(device), fixations.to(device)
            # pred_map = model(img_1)
            pred_map = model(img_1,img_2)
            # validation steps (calculating loss, etc.)
            val_loss += loss_func(pred_map, gt, fixations, args).item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch + 1} - Validation loss: {avg_val_loss:.4f}")

    # Learning rate scheduler step
    scheduler.step(avg_val_loss)

    # Early stopping check
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        counter = 0
        print(f"[{epoch + 1}, save, {args.model_saved_path}]")
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), args.model_saved_path)
        else:
            torch.save(model.state_dict(), args.model_saved_path)
    else:
        counter += 1
        print(f"No improvement for {counter} epochs!")
        if counter >= patience:
            print("Early stopping!")
            early_stop = True
            break

    print(f"Finished epoch {epoch + 1}/{args.no_epochs}")
    print("-" * 50)

# Save the final model
torch.save(model.state_dict(), args.model_saved_path)

