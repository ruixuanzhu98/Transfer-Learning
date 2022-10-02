import numpy as np
from PIL import Image

def condition(algo_num,img,alpha,beta):
    """"
    filtering script for pretraining dataset
    change with different parameter alpha and beta
    This script provides two algorithms 
    algorithm1: parameter alpha and beta
    algorithm2: only parameter alpha
    """""
    
    if algo_num==1:
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

    

