from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import image

#####plot bar chart for tissue class in pretraining phase:
df1=pd.read_csv('/dh-projects/ag-ishaque/analysis/projects/myproject/keep_1_5000_1200_file_info.csv')
tissue_name=df1['label'].unique().tolist()
tissue_num=[len(df1[df1['label']==i]) for i in tissue_name]
plt.figure(figsize=(11,8))
plt.bar(tissue_name,tissue_num)
plt.xlabel('No. of Tissue Class')
plt.ylabel('Number of Images')
plt.savefig('/dh-projects/ag-ishaque/analysis/projects/myproject/tissue_bar.jpg',bbox_inches='tight', pad_inches=0)
plt.close('all')


#####plot accurancy for training and validation in pretraining phase:
df2=pd.read_csv('/dh-projects/ag-ishaque/analysis/projects/myproject/acc_train.csv')
df3=pd.read_csv('/dh-projects/ag-ishaque/analysis/projects/myproject/acc_val.csv')
plt.subplot(1, 2, 1)
plt.plot(df2['Step'], df2['Value'],'o-')
plt.xlabel('No. of Epoch')
plt.title('Accurancy during Training')

plt.subplot(1, 2, 2)
plt.plot(df3['Step'], df3['Value'],'o-')
plt.xlabel('No. of Epoch')
plt.title('Accurancy during Validation')

plt.savefig('/dh-projects/ag-ishaque/analysis/projects/myproject/acc_train_and_val.jpg')
plt.close('all')



#####plot loss for training and validation in pretraining phase:
df4=pd.read_csv('/dh-projects/ag-ishaque/analysis/projects/myproject/loss_train.csv')
df5=pd.read_csv('/dh-projects/ag-ishaque/analysis/projects/myproject/loss_val.csv')
plt.figure(figsize=(11,8))
plt.plot(df4['Step'], df4['Value'],'.-',label='training')
plt.plot(df5['Step'], df5['Value'],'.-',label='validation')
plt.xlabel('No. of Epoch')
plt.ylabel('Loss')
plt.title('Loss during Training and Validation')
plt.legend()
plt.savefig('/dh-projects/ag-ishaque/analysis/projects/myproject/loss_train_and_val.jpg',bbox_inches='tight', pad_inches=0)
plt.close('all')



#####plot bar chart for gene class in retraining phase:
df6=pd.read_csv('/dh-projects/ag-ishaque/analysis/projects/myproject/coordinates-rescaled.csv')
gene_name=df6['gene'].unique().tolist()
gene_num=[len(df6[df6['gene']==i]) for i in gene_name]
plt.figure(figsize=(20,20))
plt.barh(gene_name,gene_num)
plt.ylabel('Name of Gene Class')
plt.xlabel('Number of Molecules')
plt.savefig('/dh-projects/ag-ishaque/analysis/projects/myproject/gene_bar.jpg',bbox_inches='tight', pad_inches=0)
plt.close('all')


#####plot numer of molecules versus validation accurancy [slice in y axis] in retraining phase:
df_y=pd.read_csv('/dh-projects/ag-ishaque/analysis/projects/myproject/test_y_axis.csv')
gene_name=df_y['gene_name'].tolist()
molecule_num=df_y['num_molecules'].tolist()
df_acc=pd.read_csv('/dh-projects/ag-ishaque/analysis/projects/myproject/f1_val_data.txt',sep=" ")
df_acc.columns=['epoch','gene_no','gene_name','loss','acc']
df_acc=df_acc[df_acc['epoch']=="epoch10"]
gene_acc=[df_acc[df_acc['gene_name']==i].iloc[0,4] for i in gene_name]
df_new=pd.DataFrame(list(zip(gene_name,gene_acc)),columns=['name','acc'])
df_new.to_csv('/dh-projects/ag-ishaque/analysis/projects/myproject/gene_acc_y_axis.csv')
plt.figure(figsize=(11,8))
plt.plot(molecule_num,gene_acc,'o-')
for i,name in enumerate(gene_name):
    plt.annotate(name,(molecule_num[i],gene_acc[i]),textcoords="offset points",xytext=(0,10),ha='center')
plt.xlabel('Name of Molecules')
plt.ylabel('Validation Accurancy')
plt.title('Numer of Molecules versus Validation Accurancy [Slice in Y Axis]')
plt.savefig('/dh-projects/ag-ishaque/analysis/projects/myproject/molecule_vs_acc_y_axis.jpg',bbox_inches='tight', pad_inches=0)
plt.close('all')
###Take a closer look at lowly-expressed genes:
plt.figure(figsize=(11,8))
plt.plot(molecule_num[:-20],gene_acc[:-20],'o-')
for i,name in enumerate(gene_name[:-20]):
    plt.annotate(name,(molecule_num[i],gene_acc[i]),textcoords="offset points",xytext=(0,10),ha='center')
plt.xlabel('Name of Molecules')
plt.ylabel('Validation Accurancy')
plt.title('Numer of Molecules versus Validation Accurancy [Slice in Y Axis]')
plt.savefig('/dh-projects/ag-ishaque/analysis/projects/myproject/molecule_vs_acc_y_axis_small.jpg',bbox_inches='tight', pad_inches=0)
plt.close('all')

#####plot numer of molecules versus validation accurancy [slice in x axis] in retraining phase:
df_x=pd.read_csv('/dh-projects/ag-ishaque/analysis/projects/myproject/test_x_axis.csv')
gene_name=df_x['gene_name'].tolist()
molecule_num=df_x['num_molecules'].tolist()
df_acc=pd.read_csv('/Users/piglet/Desktop/myproject/f2_val_data.txt',sep=" ")
df_acc.columns=['epoch','gene_no','gene_name','loss','acc']
df_acc=df_acc[df_acc['epoch']=="epoch10"]
gene_acc=[df_acc[df_acc['gene_name']==i].iloc[0,4] for i in gene_name]
df_new=pd.DataFrame(list(zip(gene_name,gene_acc)),columns=['name','acc'])
df_new.to_csv('/dh-projects/ag-ishaque/analysis/projects/myproject/gene_acc_x_axis.csv')
plt.figure(figsize=(11,8))
plt.plot(molecule_num,gene_acc,'o-')
for i,name in enumerate(gene_name):
    plt.annotate(name,(molecule_num[i],gene_acc[i]),textcoords="offset points",xytext=(0,10),ha='center')
plt.xlabel('Name of Molecules')
plt.ylabel('Validation Accurancy')
plt.title('Numer of Molecules versus Validation Accurancy [Slice in X Axis]')
plt.savefig('/dh-projects/ag-ishaque/analysis/projects/myproject/molecule_vs_acc_x_axis.jpg',bbox_inches='tight', pad_inches=0)
plt.close('all')
###Take a closer look at lowly-expressed genes:
plt.figure(figsize=(11,8))
plt.plot(molecule_num[:-20],gene_acc[:-20],'o-')
for i,name in enumerate(gene_name[:-20]):
    plt.annotate(name,(molecule_num[i],gene_acc[i]),textcoords="offset points",xytext=(0,10),ha='center')
plt.xlabel('Name of Molecules')
plt.ylabel('Validation Accurancy')
plt.title('Numer of Molecules versus Validation Accurancy [Slice in X Axis]')
plt.savefig('/dh-projects/ag-ishaque/analysis/projects/myproject/molecule_vs_acc_x_axis_small.jpg',bbox_inches='tight', pad_inches=0)
plt.close('all')