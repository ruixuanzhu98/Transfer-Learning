##### ##### jupyter notebook - Pretaining phase with H&E Staining data from Human Protein Atlas Server ##### #####

%matplotlib inline
import pandas as pd
from pathlib import Path
from glob import glob
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from collections import Counter
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader
import skimage.io as sk
from sklearn.model_selection import StratifiedKFold


### parameter dict in this file
params = {
    'model_name': 'resnet101',
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'batch_size': 32,
    'num_workers': 7,
    'num_epochs': 100,
}


#####data processing phase:
root='/dh-projects/ag-ishaque/analysis/zhur/keep_5000_1200' # variable
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
        
df=pd.DataFrame([file_list,label_list],columns=['file_path','label'])
df.to_csv(root+'file_info.csv',index=False)        


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
        label = self.data_frame.iloc[idx,1]
        sample = {'image': images, 'lanbel': label}
        if self.Transform:
            images = self.Transform(images)
        return sample

normalize=transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
data_transforms={'train':
                transforms.Compose([
                            transforms.Resize((224,224)),
                            transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize
                ]),
                'validation':
                transforms.Compose([
                            transforms.Resize((224,224)),
                            transforms.ToTensor(),
                            normalize
                ]),
}


def balancing_sampler(data_dict):
    train_labels=[label[1] for label in data_dict['train'].images]
    num_class=len(Counter(train_labels.keys()))
    labels_count=list(Counter(train_labels).values)
    labels_count=np.array(labels_count)
    weights=1.0/torch.tensor(labels_count, dtype=torch.float)
    sample_weights=weights[train_labels]
    sampler=WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    return num_class,sampler


def balancing_load(data_dict,sampler):
    loaded_data = {'train': 
                DataLoader(data_dict['train'],batch_size=params['batch_size'],samler=sampler,num_workers=params['num_workers']),
                'validation':
                DataLoader(data_dict['validation'],batch_size=params['batch_size'], num_workers=params['num_workers']) 
    }
    return loaded_data
    


def model_load(num_class):
    model = getattr(models, params['model_name'])(pretrained=True).to(params['device'])
    model.fc = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_class)).to(params['device'])
    for param in model.parameters():
        param.requires_grad = True
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    return model,criterion,optimizer



#####model training phase:
def early_stopping(self,curr_epoch,curr_loss,min_delta,patience):
    counter = 0
    if curr_epoch==0:
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
        
def model_train(loaded_data,model,criterion,optimizer,num_epochs):
    epoch_loss_train=[]
    epoch_loss_val=[]
    epoch_acc_train=[]
    epoch_acc_val=[]
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        for phase in ['train','validation']: 
            if phase == 'train':
                model.train()
            else:
                model.eval()
            current_loss=0
            current_corr_pred=0
            for inputs, labels in loaded_data[phase]:
                inputs = inputs.to(params['device'])
                labels = labels.to(params['device'])
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                _, pred=torch.max(outputs, dim=1)
                current_loss=current_loss+loss.item()*inputs.size(dim=0)
                current_corr_pred=current_corr_pred+torch.sum(pred==labels.data)
            if phase=='train':
                epoch_loss_train.append(current_loss / len(loaded_data[phase].dataset))
                epoch_acc_train.append(current_corr_pred.double() / len(loaded_data[phase].dataset))
                print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,epoch_loss_train[-1],epoch_acc_train))
            else:     
                epoch_loss_val.append(current_loss / len(loaded_data[phase].dataset))
                epoch_acc_val.append(current_corr_pred.double() / len(loaded_data[phase].dataset))
                print('{} loss: {:.4f}, acc: {:.4f}\n'.format(phase,epoch_loss_val[-1],epoch_acc_val))
                if early_stopping(epoch,epoch_loss_val[-1],min_delta=1e-4,patience=10):
                    print('\n early stopping at epoch {}\n'.format(epoch))
                    break
    return model,epoch_loss_train,epoch_loss_val,epoch_acc_train,epoch_acc_val


#####Cross Validation phase:
skf = StratifiedKFold(n_splits=5)
target = df.loc[:,'label']
fold_no = 1
model_list=[]
train_loss=[]
val_loss=[]
train_acc=[]
val_acc=[]
for train_index, val_index in skf.split(df, target):
    train = df.loc[train_index,:]
    val = df.loc[val_index,:]
    train_path=root + 'train_fold_' + str(fold_no) + '.csv'
    val_path=root + 'val_fold_' + str(fold_no) + '.csv'
    train.to_csv(train_path)
    val.to_csv(val_path)
    image_datasets = {'train':CustomDataset(root,train_path,data_transforms['train']) ,'validation':CustomDataset(root,val_path,data_transforms['validation']) }
    num_class,sampler=balancing_sampler(image_datasets)
    loaded_data=balancing_load(image_datasets,sampler)
    model,criterion,optimizer=model_load(num_class)
    model,epoch_loss_train,epoch_loss_val,epoch_acc_train,epoch_acc_val=model_train(loaded_data,model,criterion,optimizer,num_epochs=params['num_epochs'])
    model_list.append(model)
    train_loss.append(epoch_loss_train)
    val_loss.append(epoch_loss_val)
    train_acc.append(epoch_acc_train)
    val_acc.append(epoch_acc_val)
    fold_no = fold_no+1
   