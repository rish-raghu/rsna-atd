import os
import argparse
import logging
import time

import torch
from torch import nn
import numpy as np
import pandas as pd

import models
import dataset
import utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

OUTPUT_COLUMNS = [
    'patient_id', 'series_id', 
    'bowel_healthy', 'bowel_injury', 
    'extravasation_healthy', 'extravasation_injury', 
    'kidney_healthy', 'kidney_low', 'kidney_high', 
    'liver_healthy', 'liver_low', 'liver_high', 
    'spleen_healthy', 'spleen_low', 'spleen_high'
]

def evaluate(args, model, dataloader):
    predDf = torch.zeros((len(dataloader.dataset), 15)).to(device)
    b = args.batch_size

    model.eval()
    with torch.no_grad():        
        for i, data in enumerate(dataloader):
            imgs, ids = data[0].to(device), data[1].to(device)
            preds = model(imgs)
            
            predDf[i*b:(i+1)*b, 0:2] = ids
            predDf[i*b:(i+1)*b, 2:4] = nn.functional.softmax(preds[0], dim=1)
            predDf[i*b:(i+1)*b, 4:6] = nn.functional.softmax(preds[1], dim=1)
            predDf[i*b:(i+1)*b, 6:9] = nn.functional.softmax(preds[2], dim=1)
            predDf[i*b:(i+1)*b, 9:12] = nn.functional.softmax(preds[3], dim=1)
            predDf[i*b:(i+1)*b, 12:15] = nn.functional.softmax(preds[4], dim=1)
            
            logger.info("Done with {} scans".format((i+1)*b))
        
    predDf = pd.DataFrame(predDf.cpu().numpy(), columns=OUTPUT_COLUMNS)
    predDf.to_csv(args.o)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train segmentation model on images")
    parser.add_argument('arch', type=str, default='unet2d', help='unet2d, deeplabv3 (resnet50)')
    parser.add_argument('checkpoint', type=str, help="Path to model checkpoint")
    parser.add_argument('--batch-size', type=int, default=8, help="Number of images per batch")
    parser.add_argument('-o', required=True, help="Output csv file")
    args = parser.parse_args()
    logger.info(args)

    start = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    dataloader = dataset.getDataloader(args.batch_size, train=False)
    model = models.UNet2D(32).to(device)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("Loaded weights from epoch {} of {}".format(checkpoint['epoch']+1, args.checkpoint))

    evaluate(args, model, dataloader)

    logger.info(f"Total Time: {(time.time()-start):.0f} s")
