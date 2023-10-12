import wandb
import pandas as pd
import numpy as np
from data.skin_dataset import get_transform_skin
from train import run_eval
from sklearn.metrics import confusion_matrix, roc_auc_score, balanced_accuracy_score, precision_score, recall_score
from dataset_loader import CSVDatasetWithName, CSVDataset, CSVDatasetWithKeypoints
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import sar
from sam import SAM
import math
import argparse
import timm
import torchvision.transforms as transforms
import random
from algorithms_resnet_keypoints import T3A, TentFull, TentClf, TTActivation
import torchvision
import matplotlib.pyplot as plt
import csv
def load_model(model, pretrained, device):
    weights = torch.load(pretrained, map_location=device)
    model.load_state_dict(weights['model'], strict=False)
    
class AugmentOnTest:
    def __init__(self, dataset, n):
        self.dataset = dataset
        self.n = n

    def __len__(self):
        return self.n * len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i // self.n]


def run_eval(model, loader, args, test_aug=False):
    
    prog_bar_loader = loader

    all_preds = []
    all_labels = []
    all_names = []
    for iteration, (batch, label, pos_coords, coords, name) in enumerate(prog_bar_loader, 0):

        x = batch.to("cuda")
        y = label.to("cuda")
        
        if 'tta' in args.eval:
            model.eval()
            with torch.no_grad():
                if 'vit' in args.arch:
                    outputs, _ = tta.forward_vit(x, torch.Tensor(pos_coords).long(), torch.Tensor(coords).long(), adapt=True)
                else:
                    outputs, _ = tta.forward(x, torch.Tensor(pos_coords).long(), torch.Tensor(coords).long(), adapt=True)
        elif "tent" in args.eval:
            outputs = tent.forward(x, adapt=True)
            
        elif "sas" in args.eval:
            outputs = adapt_model(x) # SAR
        elif "t3a" in args.eval:
            outputs = t3a.forward(x, adapt=True)
        else:
            model.eval()
            with torch.no_grad():
                outputs = model(x)

        if not test_aug:
            all_preds += list(F.softmax(outputs, dim=1).cpu().data.numpy())
            all_labels += list(y.cpu().data.numpy())
            all_names += list(name)
        else:
            all_preds += list([np.mean(F.softmax(outputs, dim=1).cpu().data.numpy(), axis=0)])
            all_labels += list([y.cpu().data.numpy()[0]])
            all_names += list([name[0]])

    # Calculate multiclass AUC
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_names = np.array(all_names)

    run_dir = wandb.run.dir
    path = run_dir + "/" + config.eval + ".csv"
    with open(path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write the header
        writer.writerow(["image", "pred", "label"])
        
        # Write the data
        for fname, pred, label in zip(all_names, all_preds[:, 1], all_labels):
            writer.writerow([fname, pred, label])
    
    # Removed for multiclass
    auc = roc_auc_score(all_labels, all_preds[:, 1])

    cm = confusion_matrix(all_labels, all_preds.argmax(axis=1))
    acc = balanced_accuracy_score(all_labels, all_preds.argmax(axis=1))
    print('confusion matrix:\n', cm)

    return auc, acc

    
parser = argparse.ArgumentParser(description='Process some integers.')

# data
parser.add_argument('--run_name', help='name of the run')
parser.add_argument('--data_root', default='/deconstructing-bias-skin-lesion/isic-archive-512/')
parser.add_argument('--arch', help='arch')
parser.add_argument('--project', help='name of the run')
parser.add_argument('--test_csv', help='test csv')
parser.add_argument('--bf', type=str, default="empty", help="bf only for logging")
parser.add_argument('--split', type=str, default="empty", help="split for logging")
parser.add_argument('--eval', type=str, default='aug', help='test algo')
parser.add_argument('--n_pos', type=int, default=20, help='test algo')
parser.add_argument('--n_neg', type=int, default=20, help='test algo')
parser.add_argument('--alpha', type=float, default=0.4, help='test algo')
parser.add_argument('--perc', type=float, default=0.1, help='test algo')
parser.add_argument('--use_mask', type=bool, default=True, help='test algo')
args = parser.parse_args()

shuffle = False
data_sampler = None
num_workers = 8

results = {}
project_name = args.project

wandb.init(name=args.run_name, project=project_name, entity='alceubissoto', save_code=True, settings=wandb.Settings(start_method="fork"))

config = wandb.config
config.update(args)
norm = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) if 'vit' in args.arch else ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])



class RandomHorizontalFlipTransform:
    """Randomly flip the sample (image and keypoints) horizontally."""
    def __call__(self, sample):
        img, keypoints_pos, keypoints_neg = sample
        if random.random() < 0.5:
            img = transforms.functional.hflip(img)
            #mask = transforms.functional.hflip(mask)
            keypoints_pos['x_coord'] = [223 - x for x in keypoints_pos['x_coord']]
                
            if isinstance(keypoints_neg['x_coord'], list):
                keypoints_neg['x_coord'] = [223 - x for x in keypoints_neg['x_coord']]
            else:
                keypoints_neg['x_coord'] = 223 - keypoints_neg['x_coord']
                
            
        return img, keypoints_pos, keypoints_neg


class RandomVerticalFlipTransform:
    """Randomly flip the sample (image and keypoints) vertically."""
    def __call__(self, sample):
        img, keypoints_pos, keypoints_neg = sample
        if random.random() < 0.5:
            img = transforms.functional.vflip(img)
            #mask = transforms.functional.vflip(mask)

            keypoints_pos['y_coord'] = [223 - y for y in keypoints_pos['y_coord']]

                
            if isinstance(keypoints_neg['y_coord'], list):
                keypoints_neg['y_coord'] = [223 - y for y in keypoints_neg['y_coord']]
            else:
                keypoints_neg['y_coord'] = 223 - keypoints_neg['y_coord']
        return img, keypoints_pos, keypoints_neg

class ImageTransformMask:
    """Apply a transform only to the image part of a sample."""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        img, mask, keypoints = sample
        return self.transform(img), self.transform(mask), keypoints

    
class ImageTransform:
    """Apply a transform only to the image part of a sample."""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        img, mask, keypoints = sample
        return self.transform(img), mask, keypoints

# Define your transforms
transform_keypoints = transforms.Compose([
    RandomHorizontalFlipTransform(),
    RandomVerticalFlipTransform(),
    ImageTransform(transforms.RandomResizedCrop(224, scale=(1.0, 1.0))),
    ImageTransform(transforms.ColorJitter(hue=0.1)),
    ImageTransform(transforms.ToTensor()),
    ImageTransform(transforms.Normalize(norm[0],norm[1])),
    #TransformMask(transforms.Normalize([0.0], [1.0]))
])

if args.arch in ['resnet50', 'resnet50-gdro']:
    
    model_path = "./saves/{}_final_BEST_{}__{}.pth".format(project_name.split("_")[0], project_name.split("_")[0], args.run_name)
    print("Loading model_path {}".format(model_path))
    model = torchvision.models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(2048, 2) 
    load_model(model, model_path, torch.device("cuda"))

    def masked_forward(model, mask, x):
        # See note [TorchScript super()]
        x = torch.scatter(x,1, mask, 0)
        #x[:, mask, :, :] = 0.0 #mask=[B, C] x=[B, C, 7, 7] -> [B, B, C, 7, 7]
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        #x[:, mask] = 0.0 # TO-DO
        x = model.fc(x)

        return x
    
    def extract_cam(model, x): 
        # See note [TorchScript super()]
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)

        return x

elif args.arch == 'resnet50_gn':
    model = timm.create_model('resnet50_gn', pretrained=False)   
    model.fc = torch.nn.Linear(2048, 2)
    model_path = "./saves/resnetgn-final_BEST_skintrap__{}.pth".format(args.run_name)
    load_model(model, model_path, torch.device("cuda"))
    
    def masked_forward(model, mask, x):
        # See note [TorchScript super()]
        #print(x, mask)
        #x.scatter_(1, mask, 0)
        x = torch.scatter(x,1, mask, 0)
        #x[:, mask, :, :] = 0.0 #mask=[B, C] x=[B, C, 7, 7] -> [B, B, C, 7, 7]
        x = model.forward_head(x)

        return x

    def extract_cam(model, x): 
        # See note [TorchScript super()]
        x = model.forward_features(x)

        return x

elif args.arch == 'efficientnet_b0':
    model = timm.create_model('efficientnet_b0', pretrained=False)   
    model.classifier = torch.nn.Linear(model.num_features, 2)
    model_path = "./saves/efficientnet_b0-final_BEST_skintrap__{}.pth".format(args.run_name)
    load_model(model, model_path, torch.device("cuda"))
    
    def masked_forward(model, mask, x):
        # See note [TorchScript super()]
        #print(x, mask)
        #x.scatter_(1, mask, 0)
        x = torch.scatter(x,1, mask, 0)
        #x[:, mask, :, :] = 0.0 #mask=[B, C] x=[B, C, 7, 7] -> [B, B, C, 7, 7]
        x = model.forward_head(x)

        return x

    def extract_cam(model, x): 
        # See note [TorchScript super()]
        x = model.forward_features(x)

        return x
    
elif args.arch == 'vit_base_patch16_224':
    model_path = "./saves/{}_vit_base_patch16_224_final_part1_BEST_{}__{}.pth".format(project_name.split("_")[0], project_name.split("_")[0], args.run_name)
    #model_path = "./saves/vit-trap_BEST_skintrap__{}.pth".format(args.run_name)
    #model_path = "./saves/vit_base_patch16_224-final_BEST_skintrap__{}.pth".format(args.run_name)
    model = timm.create_model('vit_base_patch16_224', pretrained=False)   
    model.head = torch.nn.Linear(model.embed_dim, 2)
    load_model(model, model_path, torch.device("cuda"))
    masked_forward = None
    extract_cam = None
    
elif args.arch == 'vit_base_patch16_224.augreg_in1k':
    #model_path = "./saves/vit_base_patch16_224.augreg_in1k-final_BEST_skintrap__{}.pth".format(args.run_name)
    model_path = "./saves/vit_base_patch16_224.augreg_in1k-final_BEST_skintrap__{}.pth".format(args.run_name)
    model = timm.create_model('vit_base_patch16_224.augreg_in1k', pretrained=False)   
    model.head = torch.nn.Linear(model.embed_dim, 2)
    load_model(model, model_path, torch.device("cuda"))
    masked_forward = None
    extract_cam = None
    
model = model.to("cuda")

if "tta" in config.eval:
    hparams = {
                'mask_preprocess': 0.1,
                'mask_percentile': 0.7,
                'amt_pos': config.n_pos,
                'amt_neg': config.n_neg,
                'perc_feature_mask': config.perc,
                'alpha': config.alpha,
            }
    config.hparams = hparams
    tta = TTActivation(num_classes=2, model=model, hparams=hparams, fun_extract=extract_cam, fun_masked=masked_forward)

elif "tent" in config.eval:
    hparams = {'gamma':1, 'alpha':1, 'weight_decay':0.001, 'lr':0.001, "episodic":False}
    config.hparams = hparams
    tent = TentFull(num_classes=2, model=model, hparams=hparams)
    
elif "tentclf" in config.eval:
    hparams = {'gamma':1, 'alpha':1, 'weight_decay':0.001, 'lr':0.001, "episodic":False}
    config.hparams = hparams
    tent = TentClf(num_classes=2, model=model, hparams=hparams)

elif "t3a" in config.eval:
    hparams = {"filter_K" : 20}
    t3a = T3A(num_classes=2, model=model, hparams=hparams, fun_extract=extract_cam)
    
elif 'sas' in args.eval:
    
    lr = (0.001 / 64) if 'vit' in args.arch else (0.00025 / 64) * 1 * 2
    sar_margin_e0 = math.log(1000)*0.40
    model = sar.configure_model(model)
    params, param_names = sar.collect_params(model)
    #logger.info(param_names)

    base_optimizer = torch.optim.SGD
    optimizer = SAM(params, base_optimizer, lr=lr, momentum=0.9)
    adapt_model = sar.SAR(model, optimizer, margin_e0=sar_margin_e0) #### args.sar_margin_e0



if "noisecrop" in args.eval:
    image_dir = '/hadatasets/abissoto/isic2019-normcrop-noise/'
    ext = '.png' 
else:
    image_dir = args.data_root
    if image_dir == '/deconstructing-bias-skin-lesion/isic-archive-512/':
        ext = '.jpeg'
    else:
        ext = '.png'

test_ds = CSVDatasetWithKeypoints(image_dir, args.test_csv, 'image', 'label', transform=transform_keypoints, add_extension=ext, max_keypoints=config.n_neg, use_mask=config.use_mask)

if 'aug' in args.eval:
    REPLICAS = 50
    test_aug = True
    test_ds_replicas = AugmentOnTest(test_ds, REPLICAS)
    dl_b = DataLoader(test_ds_replicas, batch_size=REPLICAS,
                                shuffle=shuffle, num_workers=num_workers,
                                sampler=data_sampler, pin_memory=True)
else:
    dl_b = DataLoader(test_ds, batch_size=1,
                                shuffle=shuffle, num_workers=num_workers,
                                sampler=data_sampler, pin_memory=True)
    test_aug = False



test_auc, test_acc = run_eval(model, dl_b, args, test_aug=test_aug)
print(args.run_name, args.eval, "auc:", test_auc, "/ acc:", test_acc)
results['auc'] = test_auc
results['acc'] = test_acc

wandb.log(results, commit=True)
wandb.finish()



