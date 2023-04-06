# H&E staining-ISS project: Transfer  Learning  Using  Statistical  Learning  Architectures on Tissue Morphology Data

## script structure:

```
.
├── config.py
├── plot.py
├── preprocessing
│   ├── compute_mean_std.py
│   ├── filter_data.py
│   ├── filter_function.py
│   └── scraper.py
├── pretraining.py
├── readme.md
├── retrainig_x_slice
│   ├── data_x_slice.py
│   ├── retraining_x_slice.py
│   └── test_map_x_slice.py
└── retraining_y_slice
    ├── data_y_slice.py
    ├── retraining_y_slice.py
    └── test_map_y_slice.py
```

### A brief introduction:

**Please note that the paths included the scripts should be adjusted!**

***scraper.py***: sample scaper shows how to download data for one tissue class from human protein atlas server.

***config.py*** : includes variables and classes shared across the other python files; need to import this file during the running of other files.

plot.py:  plot six figures, utilizing both input and output data in pretraining and retraining phases.

compute_mean_std.py: compute mean and standard deviation of the augmented input data in the pretraining phase, used for data normalization; but also needed the calculated mean and standarad deviation in the retraining phase.

filter_function.py: includes a filter function called condition(), which is used for filtering input images both in pretraining and retraining phases; needs to be imported in filter_data.py.

filter_data.py: utilizes filter_function to choose final input images for pretraining phase and use symbolic link to new corresponding folder(each folder standards for one tissue class).

pretraining.py:  trains the resnet50 model  and afterwards get accurancy and loss during training and validation as well as the trained model 'f_trial2.pt'

**I have used two (train,test) sets for retraining phase, one (train,test) set is generated via slicing the application image at y=2300, the other one is generated via slicing at x=6000.**

data_x_slice.py/data_y_slice.py: generate postive and negative samples for each gene class in both train and test image and save all relevent information in csv files.

retraining_x_slice.py/retraining_y_slice.py: imports model 'f_trial2.pt'; freezes the first two residual blocks and trains the remaining part on the data genereated in data_x_slice.py/data_y_slice.py; gets trained model 'f1_allvsall.pt'/'f2_allvsall.pt'.

test_map_y_slice.py/test_map_y_slice.py: crops images from the entire test image and test the trained model 'f1_allvsall.pt'/'f2_allvsall.pt' on the cropped images; maps the predicted points to test images per gene class; saves the mapped image per class in folder  test_map_x_slice/test_map_y_slice.

## Order of running the above files in Terminal:

```bash
scraper.py
filter_data.py
compute_mean_std.py
pretraining.py
data_x_slice.py
data_y_slice.py
retraining_x_slice.py
retraining_y_slice.py
test_map_x_slice.py
test_map_y_slice.py
plot.py
```



​      

