#!/usr/bin/env python
import pandas as pd
from pathlib import Path
from glob2 import glob
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from collections import Counter
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from PIL import Image
import os
from config import params, AddGaussianNoise, CustomDataset_1
print('found {} gpus.'.format(torch.cuda.device_count()))
os.environ['CUDA_VISIBLE_DEVICES']="0,1,2,3,4,5,6,7"


 #####data processing phase:
root='/dh-projects/ag-ishaque/analysis/zhur/keep_1_5000_1200'
df_path='/dh-projects/ag-ishaque/analysis/zhur/keep_1_5000_1200_file_info.csv' 
df=pd.read_scv(df_path)

###calculate mean and std
cal_transform=transforms.Compose([transforms.RandomApply([transforms.RandomRotation(degrees=90)],p=0.2),
                                  transforms.ColorJitter(hue=(-0.5,0.5)),
                                  transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(contrast=1,brightness=(0.5,1.5),saturation=(0.5,1.5))]),p=0.8),
                                  transforms.RandomApply([transforms.Lambda(lambda x:F.pad(F.resize(x,size=128),128,0,"constant"))],p=prob),
                                  transforms.CenterCrop(224),
                                  transforms.RandomHorizontalFlip(p=0.5),
                                  transforms.RandomVerticalFlip(p=0.5),
                                  transforms.RandomInvert(p=0.5),
                                  transforms.RandomGrayscale(p=0.2),
                                  transforms.ToTensor()
                                 ])
def cal_mean_std(dataloader):
    batch_sum=0
    batch_sq2=0
    for images,_ in dataloader:
        images=images.to(params['device'])
        batch_sum=torch.add(batch_sum,torch.sum(images,dim=(0,2,3)))
        batch_sq2=torch.add(batch_sq2,torch.sum(torch.pow(images,2),dim=(0,2,3)))
        #inter_mean=torch.div(batch_sum,count)
        #inter_var=torch.div(batch_sq2,count)-torch.pow(inter_mean,2)
        #inter_std=torch.sqrt(inter_var)
    count=len(df)*224*224
    total_mean=torch.div(batch_sum,count)
    total_var=torch.div(batch_sq2,count)-torch.pow(total_mean,2)
    total_std=torch.sqrt(total_var)
    return total_mean,total_std


total_dataset = CustomDataset_1(root,df_path,cal_transform)
total_loader=DataLoader(total_dataset,batch_size=256,num_workers=params['num_workers'])
total_mean,total_std=cal_mean_std(total_loader)





























