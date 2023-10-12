import sys
import os
sys.path.append('..')
import os
import types
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
import h5py
import copy
from loss import LossComputer

import numpy as np
from tqdm import tqdm
#import matplotlib.pyplot as plt
#from utils import AverageMeter, accuracy
from loss import LossComputer
from dataset_loader import CSVDatasetWithName, CSVDataset, CSVDatasetWithMask, CSVDatasetWithGroups
from data.biased_dataset import BiasedDataset, BiasedDatasetWithMask
import transformers
#from pytorch_transformers import AdamW, WarmupLinearSchedule
from sklearn.metrics import confusion_matrix, roc_auc_score, balanced_accuracy_score
from data.skin_dataset import get_transform_skin #get_transform_skin_agg
from data.biased_dataset import BiasedDataset
import pickle
from torch.utils.data import WeightedRandomSampler
# imports for the tutorial
import time
import random

# pytorch
import torch.optim as optim
from torchvision.utils import make_grid
import torchvision.utils as vutils
import timm

import argparse
from algorithms_resnet_tmp import T3A, TentFull, TentClf, TTActivation
#os.environ['WANDB_MODE'] = 'dryrun'



parser = argparse.ArgumentParser(description='Process some integers.')

# data
parser.add_argument('--dataset', default='bcn')
parser.add_argument('--data_root', default='/deconstructing-bias-skin-lesion/isic-archive-512/')
parser.add_argument('--arch', default='resnet50')
parser.add_argument('--project', default='dummy')
parser.add_argument('--split', default=1, type=int, help='split')
parser.add_argument('--bf', default="0", type=str, help='split')
parser.add_argument('--num_epochs', default=100, type=int, help='amount of epochs')
parser.add_argument('--wd', default=0.001, type=float, help='amount of epochs')
parser.add_argument('--lr', default=1e-3, type=float, help='beta kl')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--robust', default=False, type=bool, help="use robust training")
parser.add_argument('--generalization_adjustment', default="0", type=str)

args = parser.parse_args()

# Setup wandb
project_name = args.project
wandb.init(project=project_name, entity='alceubissoto', save_code=True, settings=wandb.Settings(start_method="fork"))
config = wandb.config
config.update(args)
run_name = wandb.run.name

# Task characteristics
config.num_classes = 2
config.pretrained = True
    
if args.data_root == '/deconstructing-bias-skin-lesion/isic-archive-512/':
    ext = ".jpeg"
else:
    ext = ".png"

# Data loaders
train_csv = '/group_DRO/separator/trapsets/train_{}_{}_{}.csv'.format(config.dataset, config.bf, config.split)
train_set = CSVDatasetWithGroups(args.data_root, train_csv, 'image', 'label', transform=get_transform_skin(True), add_extension=ext)
if args.robust:
    group_weights = len(train_set)/train_set._group_counts
    weights = group_weights[train_set._group_array]

    # Replacement needs to be set to True, otherwise we'll run out of minority samples
    sampler = WeightedRandomSampler(weights, len(train_set), replacement=True)
    shuffle = False
else:
    group_weights = len(train_set)/train_set._y_counts
    weights = group_weights[train_set._y_array]

    # Replacement needs to be set to True, otherwise we'll run out of minority samples
    sampler = WeightedRandomSampler(weights, len(train_set), replacement=True)
    shuffle = False

train_data_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=shuffle, sampler=sampler, num_workers=8)

val_csv = '/group_DRO/separator/trapsets/val_{}_{}_{}.csv'.format(config.dataset, config.bf, config.split)
val_set = CSVDatasetWithGroups(args.data_root, val_csv, 'image', 'label', transform=get_transform_skin(False), add_extension=ext)
val_data_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=True, num_workers=8)


"""
Helper Functions
"""
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

def load_model(model, pretrained, device):
    weights = torch.load(pretrained, map_location=device)
    model.load_state_dict(weights['model'], strict=False)


def save_checkpoint(model, epoch, prefix=""):
    model_out_path = "./saves/" + project_name + '_' + prefix + '_' + run_name + ".pth"
    state = {"epoch": epoch, "model": model.state_dict()}
    if not os.path.exists("./saves/"):
        os.makedirs("./saves/")

    torch.save(state, model_out_path)

    print("model checkpoint saved @ {}".format(model_out_path))


def fine_tune_vit(dataset='cifar10', lr=2e-4, wd=1e-4, batch_size=128, num_workers=8, save_interval=5000, seed=-1, device=torch.device("cuda"), start_epoch=0, num_epochs=100,):
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        print("random seed: ", seed)

    # --------------build models -------------------------
    image_size = 224
    
    ch = 3
    best_val_perf = -1
    
    
    # ResNet50
    if config.arch == "resnet50":
        model = torchvision.models.resnet50(pretrained=config.pretrained)
        model.fc = torch.nn.Linear(2048, config.num_classes)
        
    # EfficientNet B0
    elif config.arch == "efficientnetb0":
        model = timm.create_model('efficientnet_b0', pretrained=config.pretrained)   
        model.classifier = nn.Linear(model.num_features, config.num_classes)
    
    # ViT Base 224
    elif config.arch == "vit_base_patch16_224":
        model = timm.create_model(config.arch, pretrained=True)
        model.head = nn.Linear(model.embed_dim, config.num_classes)

    model = model.to(device) 
    
    if "vit" in config.arch:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        total_steps = len(train_data_loader) * num_epochs
        warmup_steps = int(total_steps * wd) #using wd as warmup 
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        scheduler = None

    ce_criterion = torch.nn.CrossEntropyLoss(reduction="none").to(device)
    #ce_criterion = torch.nn.CrossEntropyLoss(torch.Tensor(train_set.class_weights_list).to(device), reduction="none")
    
    start_time = time.time()

    cur_iter = 0

    # process generalization adjustment stuff
    generalization_adjustment = args.generalization_adjustment
    adjustments = [float(c) for c in generalization_adjustment.split(',')]
    assert len(adjustments) in (1, train_set.n_groups)
    if len(adjustments)==1:
        adjustments = np.array(adjustments* train_set.n_groups)
    else:
        adjustments = np.array(adjustments)

    train_loss_computer = LossComputer(
        ce_criterion,
        is_robust=args.robust,
        dataset=train_set,
        alpha=0.01,
        gamma=0.1,
        adj=adjustments,
        step_size=0.01,
        normalize_loss=False,
        btl=False,
        min_var_weight=0)
 
    losses_all = AverageMeter()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, num_epochs):
        
        model.train()
        losses_all.reset()
        all_preds = []
        all_labels = []
        for iteration, (data, g) in enumerate(train_data_loader, 0):
            # --------------train------------
            
            batch = data[0]
            label = data[1]
            group = g.to(device)

            # vanilla vae training
            if len(batch.size()) == 3:
                batch = batch.unsqueeze(0)

            batch_size = batch.size(0)
            
            #mask = mask.to(device)
            real_batch = batch.to(device)
            label = label.to(device)

            # =========== Update E, D ================

            pred = model(real_batch)

            loss = train_loss_computer.loss(pred, label, group, True) # True is former "is_training". Here we only do training, so its always True
                        
            losses_all.update(loss.data.cpu())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            if scheduler is not None:
                scheduler.step()
                wandb.log({"lr":scheduler.get_lr()[0]})
      
            all_preds += list(F.softmax(pred, dim=1).cpu().data.numpy())
            all_labels += list(label.cpu().data.numpy())

        # Calculate multiclass AUC
        
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        
        if len(np.unique(all_labels)) == 2:
            auc = roc_auc_score(all_labels, all_preds[:, 1])
            wandb.log({"train_auc": auc, "epoch": epoch})

        acc = balanced_accuracy_score(all_labels, all_preds.argmax(axis=1))
        wandb.log({"train_acc": acc, "epoch": epoch})
        
        info = "\nEpoch[{}]({}/{}): time: {:4.4f}: ".format(epoch, iteration, len(train_data_loader),
                                                            time.time() - start_time)
        print(info)
        train_loss_computer.log_stats()
        train_loss_computer.reset_stats()
        wandb.log({"loss_total": losses_all.avg, "epoch": epoch})

        with torch.no_grad():
            # PERFORM VALIDATION
            all_preds = []
            all_labels = []
            for data, _ in val_data_loader:
                batch = data[0]
                label = data[1]

                model.eval()
                #mask = mask.to(device)
                real_batch = batch.to(device)
                label = label.to(device)

                outputs = model(real_batch)


                all_preds += list(F.softmax(outputs, dim=1).cpu().data.numpy())
                all_labels += list(label.cpu().data.numpy())

            # Calculate multiclass AUC

            all_labels = np.array(all_labels)
            all_preds = np.array(all_preds)

            
            if len(np.unique(all_labels)) <= 2:
                auc = roc_auc_score(all_labels, all_preds[:, 1])
                wandb.log({"val_auc": auc, "epoch": epoch})
                print('[VAL] Epoch: {} | Val AUC: {}'.format(epoch, auc))

            #cm = confusion_matrix(all_labels, all_preds.argmax(axis=1))
            #cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            #acc = np.trace(cmn) / cmn.shape[0]
            acc = balanced_accuracy_score(all_labels, all_preds.argmax(axis=1))
            wandb.log({"val_acc": acc, "epoch": epoch})
            print('[VAL] Epoch: {} | Val ACC: {}'.format(epoch, acc))   
            
            # save models
            #if len(np.unique(all_labels)) <= 2:
            #    if auc > best_val_perf:
            #        best_val_perf = auc
            #        prefix = 'BEST_' + dataset + '_'
            #        save_checkpoint(model, epoch, prefix)
            #else:
            if acc > best_val_perf:
                best_val_perf = acc
                prefix = 'BEST_' + dataset + '_'
                save_checkpoint(model, epoch, prefix)

    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print("device:", device)

model = fine_tune_vit(dataset=config.dataset, lr=config.lr, wd=config.wd, batch_size=config.batch_size, num_workers=8, seed=-1, device=device, start_epoch=0, num_epochs=config.num_epochs)

wandb.finish()
