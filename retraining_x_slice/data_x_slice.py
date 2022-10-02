#!/usr/bin/env python
#####Cropping application data for training step with dataframes:
from PIL import Image
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter
import os
from filter_function import condition

df_raw=pd.read_csv('/dh-projects/ag-ishaque/analysis/projects/HE_tran-ISS/application-data/coordinates-rescaled.csv')
df_new=df_raw.copy()
#####exchange x axis and y axis
df_new.rename(columns={'X':'Y','Y':'X'},inplace=True)
#drop the index column
df_new.drop(df_new.columns[[0]], axis=1, inplace=True)
app_image=Image.open('/dh-projects/ag-ishaque/analysis/projects/HE_tran-ISS/application-data/102ks_bg_mock_HnE.jpg')
size=[d for d in app_image.size]
gene_num=df_raw.nunique()
#####divide the application image into train data and validation data:
train_img=app_image.crop((0,0,6000,size[1]))
val_img=app_image.crop((6001,0,size[0],size[1]))
##### create molecule dataframes for each section:
index_names=df_new[(df_new['Y']>size[1])|(df_new['X']>size[0])].index
df_new.drop(index_names,inplace=True)
df_new_train=df_new.copy()
index_names=df_new_train[df_new_train['X']>6000].index
df_new_train.drop(index_names,inplace=True)
df_new_val=df_new.copy()
index_names=df_new_val[df_new_val['X']<6001].index
df_new_val.drop(index_names,inplace=True)
gene_name=df_new['gene'].unique().tolist()



def forming_folder(i,df,config):
    i_=i
    path_list=[]
    label_list=[]
    if '/' in i:
        i_=i.replace("/","_")
    if config=='train':
        path='/home/zhur/train_10/'
        array=np.zeros((6000,size[1]))
        xbound=[int(224/2),int(6000-224/2)]
        ybound=[int(224/2),int(size[1]-224/2)]
    elif config=='val':
        path='/home/zhur/val_10/'
        array=np.zeros((size[0]-6000,size[1]))
        xbound=[int(6001+224/2),int(size[0]-224/2)]
        ybound=[int(224/2),int(size[1]-224/2)]
    else:
        raise ValueError('wrong input value for config')
        exit(1)

    n_molecules=len(df[df['gene']==i])
    n_samples = n_molecules*4
    df_new=df.copy()
    index_names=df_new[df_new['gene']==i].index
    if config=='val':
    #change molecule y exis to fit the array
        df_new['X']=df_new['X']-6001
    molecules_ids=df_new.loc[index_names,['X','Y']].to_numpy().astype(int)
    array[tuple(molecules_ids.T)] = 1
    array2=array
    #initial array for negative ones

    #####sampling and saving for positive ones
    #parameter for gaussian filter
    sigma=2
    array=gaussian_filter(array,sigma)
    array[array>0]=1

    #create probability distribution over the image array
    total=np.sum(array)
    prob_dis=np.divide(array.flatten(),total).tolist()
    length=array.shape[0]*array.shape[1]
    samples= np.random.choice(np.arange(length),n_samples,replace=False,p=prob_dis)
    x_samples=samples // array.shape[1]
    y_samples=samples %  array.shape[1]
    if config=='val':
        x_samples=x_samples+6001
    for j in range(len(x_samples)):
        if (x_samples[j]<xbound[0]) or (x_samples[j]>xbound[1]) or (y_samples[j]<ybound[0]) or (y_samples[j]>ybound[1]):
            continue
        else:
            app_crop=app_image.crop((x_samples[j]-int(224/2),y_samples[j]-int(224/2),x_samples[j]+int(224/2),y_samples[j]+int(224/2)))
            new_path=path+'{}_p_{}.jpg'.format(i_,j)
            app_crop.save(new_path)
            path_list.append(new_path)
            label=1
            label_list.append(label)
    if len(label_list)==0:
        sign=False
    else:
        sign=True
        positive_len=len(label_list)
    ##### sampling and saving for negative ones
    #create probability distribution over the image array
    array2=gaussian_filter(array2,20)
    array2[array2>0]=1
    neg_array=1-array2

    total=np.sum(neg_array)
    prob_dis=np.divide(neg_array.flatten(),total).tolist()
    length=neg_array.shape[0]*neg_array.shape[1]

    samples= np.random.choice(np.arange(length),3*n_samples,replace=False,p=prob_dis)
    x_samples=samples // neg_array.shape[1]
    y_samples=samples %  neg_array.shape[1]
    if config=='val':
        x_samples=x_samples+6001
    count=0
    for j in range(len(x_samples)):
        if (x_samples[j]<xbound[0]) or (x_samples[j]>xbound[1]) or (y_samples[j]<ybound[0]) or (y_samples[j]>ybound[1]):
            continue
        else:
            app_crop=app_image.crop((x_samples[j]-int(224/2),y_samples[j]-int(224/2),x_samples[j]+int(224/2),y_samples[j]+int(224/2)))
            if condition(1,app_crop,5000,1500):
                new_path=path+'{}_n_{}.jpg'.format(i_,j)
                app_crop.save(new_path)
                label=0
                path_list.append(new_path)
                label_list.append(label)
                count=count+1
                if count>=positive_len:
                    break

    df_info=pd.DataFrame(list(zip(path_list,label_list)),columns=['file_path','label'])
    df_info_path='/home/zhur/retraining3/{}_{}_file_info.csv'.format(i_,config)
    df_info.to_csv(df_info_path,index=False)
    return sign

for i in gene_name:
    if (i not in df_new_train['gene'].tolist()) or (len(df_new_train[df_new_train['gene']==i])==1) or\
            (i not in df_new_val['gene'].tolist()):
        print('not'+i)
        continue
    else:
        i_=i
        if '/' in i:
            i_=i.replace("/","_")
        sign=forming_folder(i,df_new_val,'val')
        if sign==False:
            continue
        forming_folder(i,df_new_train,'train')
