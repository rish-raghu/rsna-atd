import logging
import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import pydicom
import cv2

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

CROP_SIZE = 512
MAX_Z = 500
IMAGE_DIR = 'train_pngs'

def getDataloader(args, split, patients=None):
    labelPath = 'data/train.csv' if split in ['train', 'val'] else None    
    if args.dataset_type=='2dsampled':
        dataset = DicomDatasetSliceSampler('data/train_series_meta.csv', args.z_size, 0.5, args.image_size, labelPath=labelPath, patientsPath=patients)
    elif args.dataset_type in ['2d', '3d']:
        dataset = DicomDataset3D('data/train_series_meta.csv', args.dataset_type, args.image_size, args.z_size, labelPath=labelPath, patientsPath=patients)
    logger.info(f"{split} split: {len(dataset)} scans")
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=(split=='train'), num_workers=args.num_workers)

def readImage(path):
    if path.endswith('.png'):
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        return pydicom.dcmread(path).pixel_array.astype(np.int32)

def _getLabel(labels, patient):
    label = labels.loc[labels['patient_id'] == patient]
    bowel = label[['bowel_healthy', 'bowel_injury']].squeeze().argmax()
    extrav = label[['extravasation_healthy', 'extravasation_injury']].squeeze().argmax()
    kidney = label[['kidney_healthy', 'kidney_low', 'kidney_high']].squeeze().argmax()
    liver = label[['liver_healthy', 'liver_low', 'liver_high']].squeeze().argmax()
    spleen = label[['spleen_healthy', 'spleen_low', 'spleen_high']].squeeze().argmax()
    return bowel, extrav, kidney, liver, spleen


class DicomDataset3D(Dataset):
    def __init__(self, metaPath, datasetType, imageSize, zSize, labelPath=None, patientsPath=None):
        self.labels = pd.read_csv(labelPath) if labelPath else None # for training
        self.meta = pd.read_csv(metaPath)
        if patientsPath: 
            with open(patientsPath, "r") as f:
                patients = [int(patient.strip()) for patient in f]
            if self.labels is not None: self.labels = self.labels[self.labels['patient_id'].isin(patients)]
            self.meta = self.meta[self.meta['patient_id'].isin(patients)]
        self.imageSize = imageSize
        self.zSize = zSize
        #self.cache = torch.zeros((len(self.meta), self.zSize, self.imageSize, self.imageSize))
        self.cache = [None] * len(self.meta)
        self.cacheHit = torch.zeros(len(self.meta))
        self.datasetType = datasetType


    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        patient, series = self.meta.iloc[idx][['patient_id', 'series_id']].astype(int)
        #print(f"PT {patient} S {series}", flush=True)
        if self.cacheHit[idx]:
            data3d = self.cache[idx]
        else:
            files = os.listdir(f'data/{IMAGE_DIR}/{patient}/{series}')
            data3d = torch.zeros((len(files), self.imageSize, self.imageSize))
            for i, file in enumerate(files):
                slice = readImage(f'data/{IMAGE_DIR}/{patient}/{series}/{file}')
                transform = transforms.Compose([transforms.CenterCrop((CROP_SIZE, CROP_SIZE)), transforms.Resize((self.imageSize, self.imageSize), antialias=True)])
                data3d[i, ...] = transform(torch.from_numpy(slice).unsqueeze(0))
            if self.zSize != -1: 
                data3d = torch.nn.functional.interpolate(data3d.unsqueeze(0).unsqueeze(0), size=(self.zSize, self.imageSize, self.imageSize))
                data3d = data3d.squeeze() if self.datasetType == '2d' else data3d.squeeze(0)
            elif len(files) > MAX_Z:
                data3d = torch.nn.functional.interpolate(data3d.unsqueeze(0).unsqueeze(0), size=(MAX_Z, self.imageSize, self.imageSize))
                data3d = data3d.squeeze() if self.datasetType == '2d' else data3d.squeeze(0)
            elif self.datasetType == '3d':
                data3d = data3d.unsqueeze(0)
            self.cache[idx] = data3d
            self.cacheHit[idx] = 1
        
        if self.labels is not None: # for training
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
        files = os.listdir(f'data/{IMAGE_DIR}/{patient}/{series}')
        if int(self.middleRange * len(files)) >= self.numSamples:
            start = int(len(files)/2 - (self.middleRange/2 * len(files)))
            end = int(len(files)/2 + (self.middleRange/2 * len(files)))
            samples = np.sort(np.random.choice(np.arange(start, end), self.numSamples, replace=False))
        else:
            samples = np.sort(np.random.choice(np.arange(len(files)), self.numSamples, replace=False))

        data3d = torch.zeros((self.numSamples, self.imageSize, self.imageSize))
        for i, image_idx in enumerate(samples):
            slice = imageRead(f'data/{IMAGE_DIR}/{patient}/{series}/{files[image_idx]}')
            transform = transforms.Compose([transforms.CenterCrop((CROP_SIZE, CROP_SIZE)), transforms.Resize((self.imageSize, self.imageSize), antialias=True)])
            data3d[i, ...] = transform(torch.from_numpy(slice).unsqueeze(0))
        data3d = data3d/torch.max(data3d)
        
        if self.labels is not None: # for training
            return data3d, torch.tensor(_getLabel(self.labels, patient))
        else: # for evaluation
            return data3d, torch.tensor([patient, series])
        
        

if __name__=='__main__':
    trainDataset = DicomDatasetSliceSampler('data/train.csv', 'data/train_series_meta.csv', 32, 0.5, 256)
    print(trainDataset[0])
