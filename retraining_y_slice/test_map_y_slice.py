#!/usr/bin/env python
from doctest import DocFileSuite
from lib2to3.pgen2.pgen import DFAState
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
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel as dp
from sklearn import preprocessing
from itertools import product
import numpy as np
from functools import partial
import os
from functools import partial
import gc
from glob2 import glob
import matplotlib.pyplot as plt
from matplotlib import image
os.environ['CUDA_VISIBLE_DEVICES']="0,1,2,3,4,5,6,7"
from retraining_y_slice import num_class, gene_name
from config import params, AddGaussianNoise, CustomDataset_3,data_transforms_2

savepath='/home/zhur/model/f1_allvsall.pt'

image_path='/home/zhur/test_image.jpg'
app_image=Image.open('/dh-projects/ag-ishaque/analysis/projects/HE_tran-ISS/application-data/102ks_bg_mock_HnE.jpg')
app_size=[d for d in app_image.size]
test_img=app_image.crop((0,0,app_size[0],2300))
test_img.save(image_path)
test_size=[d for d in test_img.size]
x_list=[]
y_list=[]
for x in range(112,int(test_size[0]-224/2)+1,16):
    for y in range(112,int(test_size[1]-224/2)+1,16):
        x_list.append(x)
        y_list.append(y)

df_spots=pd.DataFrame(list(zip(x_list,y_list)),columns=['X','Y'])


#####load the pretrained model
def model_load(num_class):
    model = getattr(models, params['model_name'])()
    model.fc = nn.Sequential(nn.Linear(2048, 1024),
                             nn.ReLU(inplace=True),
                             nn.Linear(1024,256),
                             nn.ReLU(inplace=True),
                             nn.Linear(256, num_class))
    model.load_state_dict(torch.load(savepath,map_location=params['device']))
    for name, param in model.named_parameters():
        param.requires_grad=False
    model=dp(model,device_ids=[3,6,5,4,2])
    #model.to(params['device'])
    return model

model=model_load(num_class)



#####dataloader for test data
image_dataset= CustomDataset_3(df_spots,image_path=image_path,transform=data_transforms_2)
loaded_data = DataLoader(image_dataset,batch_size=params['batch_size'],shuffle=True,num_workers=params['num_workers'])
del image_dataset

pred_x_list=[None]*num_class
pred_y_list=[None]*num_class
#####model testing:
model.eval()
h=0
model.to(params['device'])
for inputs,loc in loaded_data:
    inputs = inputs.to(params['device'])
    h=h+1

    with torch.no_grad():
        with torch.autocast(device_type='cuda'):
            outputs=model(inputs)
    del inputs
    if h%500==0:
        with torch.cuda.device('cuda:3'):
            gc.collect()
            torch.cuda.empty_cache()

    #torch.sigma() works element-wisely:
    pred=torch.sigmoid(outputs.float().to(params['device']))>0.5
    pred=pred.cpu().numpy()
    for j in range(num_class):
        if h==1:
            pred_x_list[j]=[]
            pred_y_list[j]=[]

        pred_j=pred[:,j]
        rows_index=np.where(pred_j==True)[0]
        if len(rows_index)>0:
            loc=np.array(loc).astype(float)
            pred_x_list[j]=pred_x_list[j]+loc[rows_index,0].tolist()
            pred_y_list[j]=pred_y_list[j]+loc[rows_index,1].tolist()




#####plot mapping expressions for each gene class:
for j in range(num_class):
    plt.figure(figsize=(11,8))
    test_image=image.imread(image_path)
    plt.imshow(test_image)
    plt.scatter(x=pred_x_list[j],y=pred_y_list[j],s=10)
    plt.savefig('/home/zhur/test_map_y_slice/{}.jpg'.format(gene_name[j]))
    #plt.show(block=False)
    #plt.pause(1)
    plt.close('all')









