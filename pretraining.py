#!/usr/bin/env python
import pandas as pd
from pathlib import Path
from glob2 import glob
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader,Subset
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from PIL import Image
from torch.nn.parallel import DataParallel as dp
import os
import sys
sys.path.insert(1,'/dh-projects/ag-ishaque/analysis/projects/myproject/script')
import gc
import torchvision.transforms.functional as F
from torch.utils.tensorboard import SummaryWriter
from skimage.util import random_noise
writer = SummaryWriter('runs/f_trial2')
os.environ['CUDA_VISIBLE_DEVICES']="0,1,2,3,4,5,6,7"
from config import params, AddGaussianNoise, CustomDataset_1,data_transforms_1


#####data processing phase:
#####generate csv for custom dataset:
root='/dh-projects/ag-ishaque/analysis/zhur/keep_1_5000_1200' # variable
file_list=[]
label_list=[]
all_dir=glob(root+'/*/',recursive=True)
for class_dir in all_dir:
    class_name=Path(class_dir).name
    all_file=glob(class_dir+'/*.jpg',recursive=True)
    for image_file in all_file:
        file_list.append(image_file)
        label_list.append(class_name)

df=pd.DataFrame(list(zip(file_list,label_list)),columns=['file_path','label'])
le = preprocessing.LabelEncoder()
y = df.loc[:,'label']
y=le.fit_transform(y)
df.loc[:,'label']=y
df_path=root+'_file_info.csv'
df.to_csv(df_path,index=False)


def balancing_sampler(df):
    train_labels=df.loc[:,'label']
    num_samples=len(train_labels)
    counts=train_labels.value_counts()
    num_class=len(counts)
    counts_index=counts.index.tolist()
    counts_num=counts.to_numpy()
    loc_list=[counts_index.index(train_labels.array[i]) for i in range(num_samples)]
    weights=1.0/torch.tensor(counts_num, dtype=torch.float)
    sample_weights=[weights[i] for i in loc_list]
    sampler=WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    return num_class,sampler

def balancing_load(data_dict,sampler):
    loaded_data = {'train':
                DataLoader(data_dict['train'],batch_size=params['batch_size'],sampler=sampler,num_workers=params['num_workers']),
                'validation':
                 DataLoader(data_dict['validation'],batch_size=params['batch_size'], num_workers=params['num_workers'])
    }
    return loaded_data


def model_load(num_class):
    model = getattr(models, params['model_name'])(weights="IMAGENET1K_V2")
    model.fc = nn.Linear(2048, num_class)
    model=dp(model,device_ids=[6,7,5,4,3,1,0,2])
    model.to(params['device'])
    for param in model.parameters():
        param.requires_grad =True
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    return model,criterion,optimizer


#####model training phase:
def early_stopping(curr_epoch,curr_loss,min_delta,patience):
    global counter
    global saved_loss
    if curr_epoch==0:
        counter=0
        saved_loss=curr_loss
        return False
    else:
        if curr_loss>=saved_loss+min_delta:
            counter=counter+1
        saved_loss=curr_loss
        if counter >= patience:
            return True
        else:
            return False

def model_train(loaded_data,model,criterion,optimizer,num_epochs,fold):
    epoch_loss_train=[]
    epoch_loss_val=[]
    epoch_acc_train=[]
    epoch_acc_val=[]
    use_amp = True
    counter=0
    for epoch in range(num_epochs):

        with open('/home/zhur/f_recording.txt', 'a') as f:
            f.write('Epoch_{}:\n'.format(epoch))
        for phase in ['train','validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            current_loss=0
            current_corr_pred=0
            h=0
            total=0
            scaler = torch.cuda.amp.GradScaler()
            for inputs, labels in loaded_data[phase]:
                inputs = inputs.to(params['device'])
                labels = labels.to(params['device'])
                h=h+1
                batch_size=labels.size(dim=0)

                if phase == 'train':
                    #outputs=model(inputs)
                    #loss=criterion(outputs,labels)
                    #del inputs
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type='cuda'):
                        outputs=model(inputs)
                        loss=criterion(outputs,labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    del inputs
                else:
                    with torch.no_grad():
                        with torch.autocast(device_type='cuda'):
                            outputs=model(inputs)
                            loss=criterion(outputs,labels)
                        del inputs
                if h%500==0:
                    gc.collect()
                    torch.cuda.empty_cache()
                _, pred=torch.max(outputs, dim=1)
                current_loss=current_loss+loss.item()*batch_size
                current_corr_pred=current_corr_pred+torch.sum(pred==labels.data)
                total=total+batch_size
                del labels
            if phase=='train':
                epoch_loss_train.append(current_loss / total)
                epoch_acc_train.append(current_corr_pred.double() / total)
                writer.add_scalar("Loss/train",current_loss / total, epoch+1)
                writer.add_scalar("Accurancy/train", current_corr_pred.double() / total, epoch+1)
                #with open('/home/zhur/c_train_data_{}.txt'.format(fold), 'a') as f:
                #    f.write('{} {} {}\n'.format(epoch+1,epoch_loss_train[-1],epoch_acc_train[-1]))
            else:
                epoch_loss_val.append(current_loss / total)
                epoch_acc_val.append(current_corr_pred.double() / total)
                writer.add_scalar("Loss/validation",current_loss / total, epoch+1)
                writer.add_scalar("Accurancy/validation", current_corr_pred.double() / total, epoch+1)
                #with open('/home/zhur/c_val_data_{}.txt'.format(fold), 'a') as f:
                #    f.write('{} {} {}\n'.format(epoch+1,epoch_loss_val[-1],epoch_acc_val[-1]))
        if early_stopping(epoch,epoch_loss_val[-1], min_delta=1e-4,patience=10):
            break

    savepath='/home/zhur/model/f_trial2.pt'
    torch.save(model.module.state_dict(),savepath)
    return model,epoch_loss_train,epoch_loss_val,epoch_acc_train,epoch_acc_val


#####Cross Validation phase:
skf = StratifiedKFold(n_splits=5)
X=df.drop('label',axis=1)
y = df.loc[:,'label']
fold_no = 1
model_list=[]
train_loss=[]
val_loss=[]
train_acc=[]
val_acc=[]


for train_index, val_index in skf.split(X,y):
    train = df.loc[train_index,:]
    val = df.loc[val_index,:]
    train_path=root + '_train_fold_' + str(fold_no) + '.csv'
    val_path=root + '_val_fold_' + str(fold_no) + '.csv'
    train.to_csv(train_path,index=False)
    val.to_csv(val_path,index=False)
    image_datasets = {'train':CustomDataset_1(root,train_path,data_transforms_1['train']),\
                      'validation':CustomDataset_1(root,val_path,data_transforms_1['validation']) }

   #train_subset=Subset(total_dataset,train_index)
   #val_subset=(total_dataset,val_index)
   #image_datasets = {'train':train_subset, 'validation':val_dataset }
    num_class,sampler=balancing_sampler(train)
    loaded_data=balancing_load(image_datasets,sampler)
    model,criterion,optimizer=model_load(num_class)
    model,epoch_loss_train,epoch_loss_val,epoch_acc_train,epoch_acc_val=model_train(loaded_data,model,\
    criterion,optimizer,params['num_epochs'],fold_no)
    model_list.append(model)
    train_loss.append(epoch_loss_train)
    val_loss.append(epoch_loss_val)
    train_acc.append(epoch_acc_train)
    val_acc.append(epoch_acc_val)
    fold_no = fold_no+1
    exit(0)

#####In this project, I did not run cross validation, but you could simply do it by removing line 219.
