import logging
import os
import time

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
import pydicom
import cv2

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

IMAGE_DIR = 'train_pngs'

def getDataloader(args, split, patients=None):
    labelsPath = '../data/image_level_labels_fullinjured.csv'
    t = time.time() 
    dataset = DicomImageDataset(args, split, labelsPath=labelsPath, patientsPath=patients)
    logger.info(f"{split} split: Loaded {len(dataset)} images in {(time.time()-t):.0f} seconds")
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=(split=='train'), num_workers=args.num_workers)

def readImage(path):
    if path.endswith('.png'):
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        return pydicom.dcmread(path).pixel_array.astype(np.int32)


class DicomImageDataset(Dataset):
    def __init__(self, args, split, labelsPath=None, patientsPath=None):
        self.split = split
        if labelsPath:
            self.labels = pd.read_csv(labelsPath)
            if patientsPath:
                with open(patientsPath, "r") as f:
                    patients = [int(patient.strip()) for patient in f]
                self.labels = self.labels[self.labels['patient_id'].isin(patients)].reset_index()
        
        self.data = np.zeros((len(self.labels), args.image_size, args.image_size), dtype=np.uint8)
        for i, row in self.labels.iterrows():
            slice = readImage(f"../data/{IMAGE_DIR}/{row['patient_id']}/{row['series_id']}/{row['instance_number']}.png")
            print(row['series_id'], slice.mean())
            transform = A.Compose([A.CenterCrop(args.crop_h, args.crop_w), A.Resize(args.image_size, args.image_size)])
            self.data[i, ...] = transform(image=slice)["image"]
            # transform = transforms.Compose([transforms.CenterCrop((args.crop_h, args.crop_w)), transforms.Resize((args.image_size, args.image_size), antialias=True)])
            # self.data[i, ...] = transform(torch.tensor(slice).unsqueeze(0))

        
        if split=='train':
            self.augmentTransforms = A.Compose([
                A.OneOrOther(
                    first = A.Compose([
                        A.RandomContrast(limit=0.2, p=1.0),
                        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, p=1.0)
                    ], p=1.0), # TODO: when p<1, untransformed images are 0-254 but transformed are 0-1, dividing by 255 beforehand leads to weird ranges
                    second = A.ToFloat(max_value=254, p=0.0)
                , p=1.0),
                ToTensorV2()
            ])
        else:
            self.augmentTransforms = A.Compose([
                A.ToFloat(max_value=254),
                ToTensorV2()
            ])


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.augmentTransforms(image=self.data[idx])["image"].type(torch.float)
        #print(img.max(), img.mean())
        if self.split in ['train', 'val']:
            return img, torch.tensor(self.labels.iloc[idx][['bowel', 'extravasation']], dtype=torch.uint8)
        else:
            return img, torch.tensor(self.labels.iloc[idx][['patient_id', 'series_id', 'instance_number']])


if __name__=='__main__':
    trainDataset = DicomDatasetSliceSampler('data/train.csv', 'data/train_series_meta.csv', 32, 0.5, 256)
    print(trainDataset[0])
