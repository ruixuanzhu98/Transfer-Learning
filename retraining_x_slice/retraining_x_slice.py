#!/usr/bin/env python
import torch
from PIL import Image
import pandas as pd
import sys
#sys.path.append("/home/zhur")
sys.path.insert(1, '/home/zhur')
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel as dp
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from itertools import product
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from functools import partial
from sklearn.model_selection import train_test_split
import os
from functools import partial
import gc
from glob2 import glob
from config import params, AddGaussianNoise, CustomDataset_2,data_transforms_2

os.environ['CUDA_VISIBLE_DEVICES']="0,1,2,3,4,5,6,7"
print('found {} gpus.'.format(torch.cuda.device_count()))

savepath='/home/zhur/model/f_trial2.pt'


#####load the pretrained model
all_info_csv=glob('/home/zhur/retraining2/*_train_file_info.csv',recursive=True)
num_class=len(all_info_csv)
gene_name=[os.path.split(path)[-1].split('_')[0] for path in all_info_csv]

index=gene_name.index('HBA1')
gene_name[index]='HBA1_2'


def replace_name(name):
    if '_' in name:
        name=name.replace("_","/")
    return name

df=pd.read_csv('/dh-projects/ag-ishaque/analysis/projects/HE_tran-ISS/application-data/coordinates-rescaled.csv')
num_molecules=[len(df[df['gene']==replace_name(name)]) for name in gene_name]

df_record=pd.DataFrame(list(zip(gene_name,num_molecules)),columns=['gene_name','num_molecules'])
df_record = df_record.sort_values('num_molecules', ascending=False)
df_record.to_csv('/home/zhur/test_x_axis.csv',index=True)


def model_load(num_class):
    model = getattr(models, params['model_name'])()
    model.fc = nn.Linear(2048,44)
    model.load_state_dict(torch.load(savepath,map_location=params['device']))
    model.fc = nn.Sequential(nn.Linear(2048, 1024),
                             nn.ReLU(inplace=True),
                             nn.Linear(1024,256),
                             nn.ReLU(inplace=True),
                             nn.Linear(256, num_class))
    for name, param in model.named_parameters():
        if ('fc' not in name) & ('layer3' not in name) & ('layer4' not in name):
            param.requires_grad=False
    model=dp(model,device_ids=[5,4,6,3,2])
    optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()))
    criterion=nn.BCEWithLogitsLoss()
    return model,criterion,optimizer

model,criterion,optimizer=model_load(num_class)



#####dataloader for retraining phase[using ImageFolder]
def find_batch(length):
    if length > 40000:
        size=32
    elif length > 100:
        size=8
    else:
        size=2
    return size

loaded_data=[]
for i in gene_name:
    image_dataset= {'train': CustomDataset_2('/home/zhur/retraining3/{}_train_file_info.csv'.format(i),transform=data_transforms_2),  \
                    'validation': CustomDataset_2('/home/zhur/retraining3/{}_val_file_info.csv'.format(i),transform=data_transforms_2)
    }

    size={'train':len(image_dataset['train']),'validation':len(image_dataset['train'])}

    dataloading = {'train':
                DataLoader(image_dataset['train'],batch_size=find_batch(size['train']),shuffle=True,num_workers=params['num_workers']),
               'validation':
                DataLoader(image_dataset['validation'],batch_size=find_batch(size['validation']),shuffle=True,num_workers=params['num_workers'])
    }
    loaded_data.append(dataloading)
    del image_dataset


#####model train:
model.to(params['device'])
for epoch in range(10):
    for j in range(num_class):
        scaler = torch.cuda.amp.GradScaler()
        for phase in ['train','validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            h=0
            total=0
            current_loss=0
            current_corr_pred=0
            for inputs, labels in loaded_data[j][phase]:
                inputs = inputs.to(params['device'])
                labels = labels.to(params['device'])
                h=h+1
                batch_size=labels.size(dim=0)
                optimizer.zero_grad()
                if phase == 'train':
                    with torch.autocast(device_type='cuda'):
                        outputs=model(inputs)
                        loss=criterion(outputs.float()[:,j],labels.float())

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                else:
                    with torch.no_grad():
                        with torch.autocast(device_type='cuda'):
                            outputs=model(inputs)
                            loss=criterion(outputs.float()[:,j],labels.float())
                del inputs
                if h%500==0:
                    with torch.cuda.device('cuda:7'):
                        gc.collect()
                        torch.cuda.empty_cache()

                pred=torch.sigmoid(outputs.float()[:,j]).to(params['device'])>0.5
                current_loss=current_loss+loss.item()*batch_size
                current_corr_pred=current_corr_pred+torch.sum(pred==labels.data)
                total=total+batch_size
                del labels
            if phase=='train':
                epoch_acc_train=current_corr_pred.double() / total
                epoch_loss_train=current_loss / total
                with open('/home/zhur/f2_train_data.txt', 'a') as f:
                    f.write('epoch{} {} {} {} {}\n'.format(epoch+1,j+1,gene_name[j],epoch_loss_train,epoch_acc_train))
            else:
                epoch_loss_val=current_loss / total
                epoch_acc_val=current_corr_pred.double() / total
                #writer.add_scalar("Loss_{}/validation_{}".format(gene_name[j],gene_name[j]),current_loss / total, epoch+1)
                with open('/home/zhur/f2_val_data.txt', 'a') as f:
                    f.write('epoch{} {} {} {} {}\n'.format(epoch+1,j+1,gene_name[j],epoch_loss_val,epoch_acc_val))


savepath='/home/zhur/model/f2_allvsall.pt'
torch.save(model.module.state_dict(),savepath)



