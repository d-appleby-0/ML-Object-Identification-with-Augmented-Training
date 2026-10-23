import os
import pydicom
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class DicomDataset(Dataset):
  # initialize training segment via custom dictionaries and/or transforms
  def __init__(self, dicom_dir, labels_dict=None, transforms=None):
    self.dicom_dir = dicom_dir
    self.img_names = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
    self.labels_dict = labels_dict
    self.transforms = transforms

  def __len__(self): 
    return len(self.img_names) # return length of dataset if needed

  def __getitem__(self, idx):
    img_name = self.img_names[idx] # index through hard drive to find .dcm file
    dicom_path = os.path.join(self.dicom_dir, img_name) # create exact directory via string concatenation

    ds = pydicom.dcmread(dicom_path)
    img_array = ds.pixel_array.astype(np.float32) # prevent integer division resulting in 0 pixels

    img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array)) * 255.0
    img_array = img_array.astype(np.uint8)

    if len(img_array.shape) == 2:
      img_array = np.stack([img_array] * 3, axis=-1) #convert array to 3D format needed for training in PyTorch

    target = {}
    if self.labels_dict and img_name in self.labels_dict:
      boxes = torch.as_tensor(self.labels_dict[img_name]['boxes'], dtype=torch.float32) # prepare for transformation
      labels = torch.as_tensor(self.labels_dict[img_name]['labels'], dtype=torch.int64)

      target['boxes'] = boxes
      target['labels'] = labels

    if self.transforms:
      img_array = self.transforms(img_array)

    return img_array, target
