import os
import argparse
import logging
import time

import torch
from torchvision.models.feature_extraction import create_feature_extractor
import numpy as np
import pandas as pd
import json

import models
import dataset
import utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

OUTPUT_COLUMNS = ['patient_id', 'series_id', 'instance_number', 'bowel', 'extravasation']

def evaluate(model, dataloader):
    allPreds = torch.zeros((len(dataloader.dataset), len(OUTPUT_COLUMNS)))
    b = dataloader.batch_size

    model.eval()
    with torch.no_grad():        
        dummyBatch = dataloader.dataset[0][0].unsqueeze(0).to(device)
        out = model(dummyBatch)
        embeds, preds = out['flatten'], out['fc']
        allEmbeds = torch.zeros((len(dataloader.dataset), embeds.shape[1]))

        for i, data in enumerate(dataloader):
            imgs, ids = data[0].to(device), data[1]
            out = model(imgs)
            embeds, preds = out['flatten'], out['fc']

            allPreds[i*b:(i+1)*b, :3] = ids
            allPreds[i*b:(i+1)*b, 3] = torch.sigmoid(preds[:, 0]).cpu()
            allPreds[i*b:(i+1)*b, 4] = torch.sigmoid(preds[:, 1]).cpu()
            allEmbeds[i*b:(i+1)*b, :] = embeds.cpu()
            
    return allPreds, allEmbeds


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Compute predictions and embeddings from trained model")
    parser.add_argument('config', type=str, help='.json file output by training run')
    parser.add_argument('checkpoint', type=str, help="Path to model checkpoint")
    parser.add_argument('--patients', type=str, help=".txt files with patient_ids")
    parser.add_argument('--batch-size', type=int, default=32, help="Number of images per batch")
    parser.add_argument('-o', required=True, help="Basename for output files")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)
    config.update(vars(args))
    args = utils.dotdict(config)
    logger.info(args)

    start = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    dataloader = dataset.getDataloader(args, 'test', patients=args.patients)
    model = models.getModel(args).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = create_feature_extractor(model, return_nodes=['flatten', 'fc'])
    logger.info("Loaded weights from epoch {} of {}".format(checkpoint['epoch']+1, args.checkpoint))

    preds, embeds = evaluate(model, dataloader)
    preds = pd.DataFrame(preds.numpy(), columns=OUTPUT_COLUMNS)
    preds = preds.sort_values(OUTPUT_COLUMNS[:3])
    preds.to_csv(args.o + ".csv", index=False)
    torch.save(embeds, args.o + ".pt")

    logger.info(f"Total Time: {(time.time()-start):.0f} s")
