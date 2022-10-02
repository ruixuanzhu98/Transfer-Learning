#!/usr/bin/env python
import numpy as np
import os
from PIL import Image
from filter_function import condition

def filter_data(tissue_name,algo_num,alpha,beta=None):
    tissue_path=os.path.join('/data/analysis/ag-ishaque/projects/HE-tran-ISS/data',tissue_name,'raw-0')
    files = os.listdir(tissue_path)
    new_path_keep='/data/analysis/ag-ishaque/zhur/keep_{}_{}_{}'.format(algo_num,alpha,beta)
    os.chdir(new_path_keep)
    new_dir1=tissue_name
    os.mkdir(new_dir1,mode=0o777)
    new_path_discard='/data/analysis/ag-ishaque/zhur/discard_{}_{}_{}'.format(algo_num,alpha,beta)
    os.chdir(new_path_discard)
    new_dir2=tissue_name
    os.mkdir(new_dir2,mode=0o777)
    n=0
    for file in files:
        img=Image.open(os.path.join(tissue_path,file))
        if condition(algo_num,img,alpha,beta):
            #img.save(os.path.join(new_path_keep,new_dir1,file),'JPEG') 
            #shutil.copyfile(os.path.join(tissue_path,file),os.path.join(new_path_keep,new_dir1,file))
            os.symlink(os.path.join(tissue_path,file),os.path.join(new_path_keep,new_dir1,file))
            n=n+1  
        else: 
            #img.save(os.path.join(new_path_discard,new_dir2,file),'JPEG')
            #shutil.copyfile(os.path.join(tissue_path,file),os.path.join(new_path_discard,new_dir2,file))
            os.symlink(os.path.join(tissue_path,file),os.path.join(new_path_discard,new_dir2,file))
    return n/len(files)


#####Implementing the filter_function for all downloaded images:
raw_data_dir=['adipose','adrenal-gland','appendix','bone-marrow','breast','bronchus','caudate','cerebellum','cervix','colon','cortex','duodenum',\
'endometrium','epididymis','esophagus','fallopian-tube','gallbladder','heart','hippocampus','kidney','liver','lung','lymph-node','nasopharynx',\
'oral-mucosa','ovary','pancreas','parathyroid','placenta','prostate','rectum','salvary-gland','seminal-vesicle','skeletal-muscle','skin',\
'small-intestine','smooth-muscle','spleen','stomach','testes','thyroid','tonsil','urinary-bladder','vagina']   
os.mkdir('keep_1_5000_1200',mode=0o777)
os.mkdir('discard_1_5000_1200',mode=0o777)
os.mkdir('keep_2_300_None',mode=0o777)
for raw_name in raw_data_dir:
    filter_data(raw_name,1,5000,1200)
    filter_data(raw_name,2,300,None) 
