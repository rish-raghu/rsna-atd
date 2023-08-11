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

def getDataloader(batch_size):
    trainDataset = DicomDatasetSliceSampler('data/train.csv', 'data/train_series_meta.csv', 32, 0.5, 256)
    logger.info("Dataset size: " + str(len(trainDataset)))
    return DataLoader(trainDataset, batch_size=batch_size, shuffle=True)


def _getLabel(labels, patient):
    label = labels.loc[labels['patient_id'] == patient]
    bowel = label[['bowel_healthy', 'bowel_injury']].squeeze().argmax()
    extrav = label[['extravasation_healthy', 'extravasation_injury']].squeeze().argmax()
    kidney = label[['kidney_healthy', 'kidney_low', 'kidney_high']].squeeze().argmax()
    liver = label[['liver_healthy', 'liver_low', 'liver_high']].squeeze().argmax()
    spleen = label[['spleen_healthy', 'spleen_low', 'spleen_high']].squeeze().argmax()
    return bowel, extrav, kidney, liver, spleen


class DicomDataset3D(Dataset):
    def __init__(self, labelPath, metaPath):
        self.labels = pd.read_csv(labelPath)
        self.meta = pd.read_csv(metaPath)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        patient, series = self.meta.iloc[idx][['patient_id', 'series_id']].astype(int)
        files = os.listdir('data/train_images/{}/{}'.format(patient, series))
        data3d = torch.zeros((len(files), 512, 512))
        for i, file in enumerate(files):
            slice = pydicom.dcmread('data/train_images/{}/{}/{}'.format(patient, series, file)).pixel_array
            data3d[i, ...] = transforms.CenterCrop(512)(torch.from_numpy(slice.astype(np.int32)))
          
        return data3d, _getLabel(self.labels, patient)


class DicomDatasetSliceSampler(Dataset):
    def __init__(self, labelPath, metaPath, numSamples, middleRange, imageSize):
        self.labels = pd.read_csv(labelPath)
        self.meta = pd.read_csv(metaPath)
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
          
        return data3d, torch.tensor(_getLabel(self.labels, patient))
        
        

if __name__=='__main__':
    trainDataset = DicomDatasetSliceSampler('data/train.csv', 'data/train_series_meta.csv', 32, 0.5, 256)
    print(trainDataset[0])
