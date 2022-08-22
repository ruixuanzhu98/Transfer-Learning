#!/usr/bin/env python
# coding: utf-8

# ## Web scraper for brochnial tissue from the HPA

import urllib.request
import os,sys,stat



out_dir='/data/analysis/ag-ishaque/projects/HE-tran-ISS/data/bronchus/raw-0'

import matplotlib.pyplot as plt
import tqdm

files = os.listdir(out_dir)

os.mkdir('/data/analysis/ag-ishaque/projects/HE-tran-ISS/data/bronchus/filtered-0')
os.chmod('/data/analysis/ag-ishaque/projects/HE-tran-ISS/data/bronchus/filtered-0',stat.S_IRWXO)


def condition(img):
    return (img[:,:,0].std()>15) & (img.shape==(256, 256, 3))


for i,file in tqdm.tqdm_notebook(enumerate(files),total=len(files)):
    img = plt.imread(os.path.join(out_dir,file))
    if condition(img):
        plt.imsave(f'/data/analysis/ag-ishaque/projects/HE-tran-ISS/data/bronchus/filtered-0/{file}.jpg',img)
    
#(((img.mean(-1)/256)<0.9).mean()>0.1
