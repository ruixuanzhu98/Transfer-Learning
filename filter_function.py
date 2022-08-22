import numpy as np
import os,sys
from PIL import Image
import shutil
def filter_data(tissue_name,algo_num,alpha,beta=None):
    """"
    filtering script for pretraining dataset
    change with different parameter alpha
    This script using R=G=B as criterion
    """
    def condition(num,img):
        if num==1:
            img_arr=np.array(img)
            rgb_criterion1=np.sum((np.abs(img_arr[:,:,0]-img_arr[:,:,1])<5) & (np.abs(img_arr[:,:,1]-img_arr[:,:,2])<5) \
                          & (np.abs(img_arr[:,:,0]-img_arr[:,:,2])<5)) 
            rgb_criterion2=img_arr[:,:,0].std()
            criterion=((rgb_criterion1<alpha) or (np.sum(img_arr[:,:,1]<200)>beta)) and (rgb_criterion2>5)  
        else:
            img2=img.convert(mode="HSV")
            img2_arr=np.array(img2)
            ###converting rgb to hsv via Pillow
            hsv_criterion=np.sum(img2_arr[:,:,1]>30)
            criterion=hsv_criterion > alpha
        return (criterion and (np.shape(img)==(256, 256, 3)))
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
        if condition(algo_num,img):
            #img.save(os.path.join(new_path_keep,new_dir1,file),'JPEG') 
            shutil.copyfile(os.path.join(tissue_path,file),os.path.join(new_path_keep,new_dir1,file))
            n=n+1  
        else: 
            #img.save(os.path.join(new_path_discard,new_dir2,file),'JPEG')
            shutil.copyfile(os.path.join(tissue_path,file),os.path.join(new_path_discard,new_dir2,file))
    return n/len(files)

