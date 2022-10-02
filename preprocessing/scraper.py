#####scraper for downloading all possible images from human protein atlas server:
##### This is an example of tissue class urinary bladder:
import urllib.request
import os
# root directory for placenta tissue image tiles:
root = 'https://images.proteinatlas.org/dictionary_images/fileup5ebaa17e379a7989345256_files/'
# requested size:
size = 17
root = root+str(size)+'/'
out_dir='/data/analysis/ag-ishaque/projects/HE-tran-ISS/data/urinary-bladder/raw-0'
x=0
while True:
    y=0
    while True:
        url=f'{root}{x}_{y}.jpg'
        try:
            urllib.request.urlretrieve(url, f'{out_dir}/{size}_{x}_{y}.jpg')
        except Exception as e: 
            x+=1
            break
        y+=1
    
    if y==0:
        break