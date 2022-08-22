import numpy as np
import os,sys
from PIL import Image
from filter_function import *

#raw_data_dir=['adipose','adrenal-gland','appendix','bone-marrow','breast','bronchus','caudate','cerebellum','cervix','colon','cortex','duodenum',\
#'endometrium','epididymis','esophagus','fallopian-tube','gallbladder','heart','hippocampus','kidney','liver','lung','lymph-node','nasopharynx',\
#'oral-mucosa','ovary','pancreas','parathyroid','placenta','prostate','rectum','salvary-gland','seminal-vesicle','skeletal-muscle','skin',\
#'small-intestine','smooth-muscle','spleen','stomach','testes','thyroid','thyroid','urinary-bladder','vagina']  
raw_data_dir=['urinary-bladder','vagina'] 
#os.mkdir('keep_1_5000_1200',mode=0o777)
#os.mkdir('discard_1_5000_1200',mode=0o777)
#os.mkdir('keep_2_300_None',mode=0o777)
for raw_name in raw_data_dir:
    filter_data(raw_name,1,5000,1200)
    filter_data(raw_name,2,300,None) 
    

