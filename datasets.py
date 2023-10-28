import collections
import csv
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.io import read_image


# TODO Task 1b - Implement LesionDataset
#        You must implement the __init__, __len__ and __getitem__ methods.
#
#        The __init__ function should have the following prototype
#          def __init__(self, img_dir, labels_fname):
#            - img_dir is the directory path with all the image files
#            - labels_fname is the csv file with image ids and their 
#              corresponding labels
#
#        Note: You should not open all the image files in your __init__.
#              Instead, just read in all the file names into a list and
#              open the required image file in the __getitem__ function.
#              This prevents the machine from running out of memory.
#
# TODO Task 1e - Add augment flag to LesionDataset, so the __init__ function
#                now look like this:
#                   def __init__(self, img_dir, labels_fname, augment=False):
#



class LesionDataset(torch.utils.data.Dataset):
  def __init__(self, img_dir, labels_fname, transform=None, augment=False):
    self.labels_fname = pd.read_csv(labels_fname)
    self.img_dir = img_dir
    self.transform = transforms.Compose([
                        transforms.RandomVerticalFlip(0.5),
                        transforms.RandomPerspective(30),
                        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.2, hue=0.05),
                        transforms.ToTensor(),
                        
    ])

  def __len__(self):
    return len(self.labels_fname)

  def __getitem__(self, idx):
    img_path = self.img_dir + '/' + f'{self.labels_fname.iloc[idx, 0]}.jpg'

    image = read_image(img_path)/255.0
#    label = torch.tensor(self.labels_fname.values[idx, 1:8].astype(np.float), dtype=torch.float32)
    label = torch.tensor(self.labels_fname.values[idx, 1:8].argmax()).long()   
    image = Image.open(img_path)

    image = self.transform(image)
#    image = torch.Tensor(np.array(image))

#    image = image.permute(2,0,1)


    return image, label