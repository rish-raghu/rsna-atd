import logging
import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
import pandas as pd
import pydicom

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

IMAGE_SIZE = 512

def getDataloader(batch_size, patientsPath=None, train=True):
    labelPath = 'data/train.csv' if train else None
    #dataset = DicomDatasetSliceSampler('data/train_series_meta.csv', 32, 0.5, 256, labelPath=labelPath)
    dataset = DicomDataset3D('data/train_series_meta.csv', 128, 128, labelPath=labelPath, patientsPath=patientsPath)
    logger.info(f"Using {len(dataset)} scans")
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)


def _getLabel(labels, patient):
    label = labels.loc[labels['patient_id'] == patient]
    bowel = label[['bowel_healthy', 'bowel_injury']].squeeze().argmax()
    extrav = label[['extravasation_healthy', 'extravasation_injury']].squeeze().argmax()
    kidney = label[['kidney_healthy', 'kidney_low', 'kidney_high']].squeeze().argmax()
    liver = label[['liver_healthy', 'liver_low', 'liver_high']].squeeze().argmax()
    spleen = label[['spleen_healthy', 'spleen_low', 'spleen_high']].squeeze().argmax()
    return bowel, extrav, kidney, liver, spleen


class DicomDataset3D(Dataset):
    def __init__(self, metaPath, imageSize, zSize, labelPath=None, patientsPath=None):
        self.labels = pd.read_csv(labelPath) if labelPath else None # for training
        self.meta = pd.read_csv(metaPath)
        if patientsPath: 
            with open(patientsPath, "r") as f:
                patients = [int(patient.strip()) for patient in f]
            self.labels = self.labels[self.labels['patient_id'].isin(patients)]
            self.meta = self.meta[self.meta['patient_id'].isin(patients)]
        self.imageSize = imageSize
        self.zSize = zSize

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        patient, series = self.meta.iloc[idx][['patient_id', 'series_id']].astype(int)
        files = os.listdir('data/train_images/{}/{}'.format(patient, series))
        data3d = torch.zeros((len(files), self.imageSize, self.imageSize))
        for i, file in enumerate(files):
            slice = pydicom.dcmread('data/train_images/{}/{}/{}'.format(patient, series, file)).pixel_array
            transform = transforms.Compose([transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)), transforms.Resize((self.imageSize, self.imageSize), antialias=True)])
            data3d[i, ...] = transform(torch.from_numpy(slice.astype(np.int32)).unsqueeze(0))
        data3d = torch.nn.functional.interpolate(data3d, size=(self.zSize, self.imageSize, self.imageSize))
        data3d = data3d/torch.max(data3d)
        
        if self.labels: # for training
            return data3d, torch.tensor(_getLabel(self.labels, patient))
        else: # for evaluation
            return data3d, torch.tensor([patient, series])


class DicomDatasetSliceSampler(Dataset):
    def __init__(self, metaPath, numSamples, middleRange, imageSize, patientsPath=None, idx=None):
        self.labels = pd.read_csv(labelPath) if labelPath else None # for training
        self.meta = pd.read_csv(metaPath)
        if patientsPath: 
            with open(patientsPath, "r") as f:
                patients = [int(patient.strip()) for patient in f]
            self.labels = self.labels[self.labels['patient_id'].isin(patients)]
            self.meta = self.meta[self.meta['patient_id'].isin(patients)]
        self.numSamples = numSamples
        self.middleRange = middleRange
        self.imageSize = imageSize

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        patient, series = self.meta.iloc[idx][['patient_id', 'series_id']].astype(int)
        files = os.listdir('data/train_images/{}/{}'.format(patient, series))
        if int(self.middleRange * len(files)) >= self.numSamples:
            start = int(len(files)/2 - (self.middleRange/2 * len(files)))
            end = int(len(files)/2 + (self.middleRange/2 * len(files)))
            samples = np.sort(np.random.choice(np.arange(start, end), self.numSamples, replace=False))
        else:
            samples = np.sort(np.random.choice(np.arange(len(files)), self.numSamples, replace=False))

        data3d = torch.zeros((self.numSamples, self.imageSize, self.imageSize))
        for i, image_idx in enumerate(samples):
            slice = pydicom.dcmread('data/train_images/{}/{}/{}'.format(patient, series, files[image_idx])).pixel_array
            transform = transforms.Compose([transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)), transforms.Resize((self.imageSize, self.imageSize), antialias=True)])
            data3d[i, ...] = transform(torch.from_numpy(slice.astype(np.int32)).unsqueeze(0))
        data3d = data3d/torch.max(data3d)
        
        if self.labels: # for training
            return data3d, torch.tensor(_getLabel(self.labels, patient))
        else: # for evaluation
            return data3d, torch.tensor([patient, series])
        
        

if __name__=='__main__':
    trainDataset = DicomDatasetSliceSampler('data/train.csv', 'data/train_series_meta.csv', 32, 0.5, 256)
    print(trainDataset[0])
