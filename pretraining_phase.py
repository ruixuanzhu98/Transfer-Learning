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
import skimage.io as sk
from torch.nn.parallel import DataParallel as dp
import os
import gc
print('found {} gpus.'.format(torch.cuda.device_count()))
os.environ['CUDA_VISIBLE_DEVICES']="0,1,2,3,4,5,6,7"

### parameter dict in this file
params = {
    'model_name': 'resnet50',
    'device': torch.device('cuda:{}'.format(4)),
    'batch_size': 128,
    'num_workers': 32,
    'num_epochs': 50,
}

#####data processing phase:
root='/dh-projects/ag-ishaque/analysis/zhur/keep_1_5000_1200' # variable
file_list=[]
label_list=[]
#df=pd.DataFrame(columns=['path','label'])
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


##define transform class to adding noise to images
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):

           self.std = std
           self.mean = mean

    def __call__(self, tensor):

           return tensor + torch.randn(tensor.size()) * self.std + self.mean



###calculate mean and std
#cal_transform=transforms.Compose([transforms.RandomApply([transforms.RandomRotation(degrees=90)],p=0.2),
 #           transforms.CenterCrop(224),
  #          transforms.RandomHorizontalFlip(p=0.5),
   #         transforms.RandomVerticalFlip(p=0.5),
    #        transforms.RandomInvert(p=0.5),
     #       transforms.RandomGrayscale(p=0.2),
      #      transforms.ToTensor()
       #       ])
#def cal_mean_std(dataloader):
    #k=0
 #   batch_sum=0
  #  batch_sq2=0
   # for images,_ in dataloader:
    #       images=images.to(params['device'])
     #      batch_sum=torch.add(batch_sum,torch.sum(images,dim=(0,2,3)))
      #     batch_sq2=torch.add(batch_sq2,torch.sum(torch.pow(images,2),dim=(0,2,3)))
           #k=k+1
           #count=k*256*224*224
           #inter_mean=torch.div(batch_sum,count)
           #inter_var=torch.div(batch_sq2,count)-torch.pow(inter_mean,2)
           #inter_std=torch.sqrt(inter_var)
           #print(inter_mean,inter_std)
  #  count=len(df)*224*224
   # total_mean=torch.div(batch_sum,count)
   # total_var=torch.div(batch_sq2,count)-torch.pow(total_mean,2)
   # total_std=torch.sqrt(total_var)
   # return total_mean,total_std


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir,csv_file, transform=None):
        self.root_dir = root_dir
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_path = self.data_frame.iloc[idx,0]
        images = sk.imread(image_path)
        labels = self.data_frame.iloc[idx,1]
        if self.transform:
            image_pil=Image.fromarray(images)
            images = self.transform(image_pil)
        #sample = {'image': images, 'label': label}
        return images,labels

#total_dataset = CustomDataset(root,df_path,cal_transform)
#total_loader=DataLoader(total_dataset,batch_size=256,num_workers=params['num_workers'])
#print('ok!')
#total_mean,total_std=cal_mean_std(total_loader)
#print(total_mean,total_std)
#with open('/home/zhur/params.txt','a') as f:
 #       f.write('{}\n {}'.format(total_mean,total_std))

#exit(0)
total_mean=[0.5000,0.5000,0.5001]
total_std=[0.3287,0.2773,0.3078]
normalize=transforms.Normalize(mean=total_mean,std=total_std)
data_transforms={'train':
                transforms.Compose([
                            transforms.RandomApply([transforms.RandomRotation(degrees=90)],p=0.2),
                            transforms.CenterCrop(224),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomVerticalFlip(p=0.5),
                            transforms.RandomInvert(p=0.5),
                            transforms.RandomGrayscale(p=0.2),
                            transforms.ToTensor(),
                            normalize,
                            transforms.RandomApply([AddGaussianNoise(0.,1.)],p=0.5)
                ]),
                'validation':
                transforms.Compose([
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize
                ]),
}


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
    model=dp(model,device_ids=[4,5,6,7,2,1,3,0])
    model.to(params['device'])
    for param in model.parameters():
        param.requires_grad =True
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    return model,criterion,optimizer


#####model training phase:
def early_stopping(curr_epoch,curr_loss,saved_loss,min_delta,patience):
    counter = 0
    if curr_loss>=saved_loss+min_delta:
        counter=counter+1
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
    for epoch in range(num_epochs):
        scaler = torch.cuda.amp.GradScaler()
        with open('/home/zhur/recording.txt', 'a') as f:
            f.write('Fold_{} Epoch_{}:\n'.format(fold,epoch))
        for phase in ['train','validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            current_loss=0
            current_corr_pred=0
            h=0
            total=0
            for inputs, labels in loaded_data[phase]:
                inputs = inputs.to(params['device'])
                labels = labels.to(params['device'])
                h=h+1
                batch_size=labels.size(dim=0)
                if h%100==0:
                    with open('/home/zhur/recording1.txt', 'a') as f:
                        f.write('{}\n'.format(h))
                    print('{}\n'.format(h))
                if phase == 'train':
                    #outputs=model(inputs)
                    #loss=criterion(outputs,labels)
                    #del inputs
                    optimizer.zero_grad()
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                        outputs=model(inputs)
                        loss=criterion(outputs,labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    del inputs
                else:
                    with torch.no_grad():
                        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
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
                with open('/home/zhur/train_data_{}.txt'.format(fold), 'a') as f:
                    f.write('{} {} {}\n'.format(epoch+1,epoch_loss_train[-1],epoch_acc_train[-1]))
            else:
                epoch_loss_val.append(current_loss / total)
                epoch_acc_val.append(current_corr_pred.double() / total)
                with open('/home/zhur/val_data_{}.txt'.format(fold), 'a') as f:
                    f.write('{} {} {}\n'.format(epoch+1,epoch_loss_val[-1],epoch_acc_val[-1]))
                if epoch!=0:
                    if early_stopping(epoch,epoch_loss_val[-1],epoch_loss_val[-2], min_delta=1e-4,patience=10):
                        break
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    savepath='/home/zhur/model/pretraining'
    torch.save(state,savepath)
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
    image_datasets = {'train':CustomDataset(root,train_path,data_transforms['train']),\
                      'validation':CustomDataset(root,val_path,data_transforms['validation']) }

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
