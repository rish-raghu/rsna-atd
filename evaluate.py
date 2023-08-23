import os
import argparse
import logging
import time

import torch
import numpy as np
import pandas as pd
import json

from models.utils import get_model
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

def evaluate(model, dataloader):
    preds = torch.zeros((len(dataloader.dataset), len(OUTPUT_COLUMNS)))
    b = dataloader.batch_size

    model.eval()
    with torch.no_grad():        
        for i, data in enumerate(dataloader):
            imgs, ids = data[0].to(device), data[1].to(device)
            probs = model(imgs)
            
            preds[i*b:(i+1)*b, 0:2] = ids.cpu()
            preds[i*b:(i+1)*b, 2:4] = torch.nn.functional.softmax(probs[0], dim=1).cpu()
            preds[i*b:(i+1)*b, 4:6] = torch.nn.functional.softmax(probs[1], dim=1).cpu()
            preds[i*b:(i+1)*b, 6:9] = torch.nn.functional.softmax(probs[2], dim=1).cpu()
            preds[i*b:(i+1)*b, 9:12] = torch.nn.functional.softmax(probs[3], dim=1).cpu()
            preds[i*b:(i+1)*b, 12:15] = torch.nn.functional.softmax(probs[4], dim=1).cpu()
            
            logger.info("Done with {} scans".format((i+1)*b))

    return preds


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train segmentation model on images")
    parser.add_argument('config', type=str, help='.json file output by training run')
    parser.add_argument('checkpoint', type=str, help="Path to model checkpoint")
    parser.add_argument('--patients', type=str, help=".txt files with patient_ids")
    parser.add_argument('--batch-size', type=int, default=8, help="Number of images per batch")
    parser.add_argument('-o', required=True, help="Output csv file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)
    config.update(vars(args))
    args = utils.dotdict(config)
    logger.info(args)

    start = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    dataloader = dataset.getDataloader(args, 'test', patients=args.patients)
    model = get_model(args).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("Loaded weights from epoch {} of {}".format(checkpoint['epoch']+1, args.checkpoint))

    preds = evaluate(model, dataloader)
    preds = pd.DataFrame(preds.cpu().numpy(), columns=OUTPUT_COLUMNS)
    preds.to_csv(args.o, index=False)

    logger.info(f"Total Time: {(time.time()-start):.0f} s")
