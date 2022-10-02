import torch
from skimage.util import random_noise
from PIL import Image
import pandas as pd
from torchvision import datasets, models, transforms
import numpy as np

### parameter dict in this project
params = {
    'model_name': 'resnet50',
    'device': torch.device('cuda:{}'.format(6)),
    'batch_size': 256,
    'num_workers': 40,
    'num_epochs': 50,
}

##define transform class to adding noise to images
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):

           self.std = std
           self.mean = mean

    def __call__(self, tensor):

           return torch.from_numpy(random_noise(tensor, mode='gaussian', mean=0, var=0.05, clip=True)).float()

#####custom dataset for pretraining phase:
class CustomDataset_1(torch.utils.data.Dataset):
    def __init__(self, root_dir,csv_file, transform=None):
        self.root_dir = root_dir
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_path = self.data_frame.iloc[idx,0]
        images = Image.open(image_path)
        labels = self.data_frame.iloc[idx,1]
        if self.transform:
            images = self.transform(images)
        #sample = {'image': images, 'label': label}
        return images,labels

#####total_mean and total_std calcalated from computes_mean_std.py
total_mean=[0.4999, 0.4999, 0.5000]
total_std=[0.3425, 0.3418, 0.3426]


#####data transforms for pretraining dataset:
data_transforms_1={'train':
                transforms.Compose([
                            transforms.RandomApply(torch.nn.ModuleList([transforms.RandomRotation(degrees=90)]),p=0.2),
                            transforms.ColorJitter(hue=(-0.5,0.5)),
                            transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(contrast=1,brightness=(0.5,1.5),saturation=(0.5,1.5))]),p=0.8),
                            transforms.RandomApply([transforms.Lambda(lambda x:F.pad(F.resize(x,size=128),128,0,"constant"))],p=0.2),
                            transforms.CenterCrop(224),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomVerticalFlip(p=0.5),
                            transforms.RandomInvert(p=0.5),
                            transforms.RandomGrayscale(p=0.2),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=total_mean,std=total_std),
                            transforms.RandomApply([AddGaussianNoise(0.,1.)],p=0.5)
                ]),
                'validation':
                transforms.Compose([
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=total_mean,std=total_std),
                ]),
}

#####custom image dataset for retraining phase:
class CustomDataset_2(torch.utils.data.Dataset):
    def __init__(self, csv_path,transform=None):
        self.data_frame = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.data_frame.iloc[idx,0]
        images = Image.open(image_path)
        labels = self.data_frame.iloc[idx,1]

        if self.transform:
            images = self.transform(images)

        return images,labels

#####data transforms for retraining dataset:
data_transforms_2=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=total_mean,std=total_std)
                ])

#####custom image dataset for mapping validation predictions 
class CustomDataset_3(torch.utils.data.Dataset):
    def __init__(self, df,image_path,transform=None):
        self.data_frame = df
        self.image_path=image_path
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x= self.data_frame.iloc[idx,0]
        y=self.data_frame.iloc[idx,1]
        test_img = Image.open(self.image_path)
        images=test_img.crop((x-int(224/2),y-int(224/2),x+int(224/2),y+int(224/2)))
        #labels = self.data_frame.iloc[idx,1]

        if self.transform:
            images = self.transform(images)
        loc=np.array([x,y])
        return images,loc
