import os
import argparse
import logging
import time
import sys
import json

import torch
import torch.nn.functional as F
import numpy as np
import wandb
from sklearn.metrics import precision_recall_fscore_support

import models
import dataset
import utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')


def _addMetrics(preds, targets, split, epochMetrics):
    precision, recall, fbeta, _ = precision_recall_fscore_support(targets[:, 0], preds[:, 0], average='binary')
    epochMetrics[f"{split}_precision_bowel"] = precision
    epochMetrics[f"{split}_recall_bowel"] = recall
    epochMetrics[f"{split}_fbeta_bowel"] = fbeta

    precision, recall, fbeta, _ = precision_recall_fscore_support(targets[:, 1], preds[:, 1], average='binary')
    epochMetrics[f"{split}_precision_extrav"] = precision
    epochMetrics[f"{split}_recall_extrav"] = recall
    epochMetrics[f"{split}_fbeta_extrav"] = fbeta


def train(args, model, dataloaders):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    lossFn = torch.nn.BCELoss()
    bestTrainLoss = float('inf')
    bestValLoss = float('inf')
    earlyStopCount = 0
    
    N = len(dataloaders['train'].dataset)
    B = len(dataloaders['train'])

    for epoch in range(args.epochs):
        totalTrainLoss = 0.0
        allPreds, allTargets = torch.zeros((N, 2)), torch.zeros((N, 2))
        epochMetrics = {}
        t = time.time()
        model.train()
        
        for i, data in enumerate(dataloaders['train']):
            imgs, targets = data[0].to(device), data[1].to(device)
            preds = torch.sigmoid(model(imgs))
            optimizer.zero_grad()

            bowelLoss = lossFn(preds[:, 0], targets[:, 0].to(torch.float))
            extravLoss = lossFn(preds[:, 1], targets[:, 1].to(torch.float))
            loss = bowelLoss + extravLoss
            loss.backward()
            optimizer.step()

            epochMetrics[f"train_loss_bowel"] = epochMetrics.get("train_loss_bowel", 0) + bowelLoss.item()/B
            epochMetrics[f"train_loss_extrav"] = epochMetrics.get("train_loss_extrav", 0) + extravLoss.item()/B
            allPreds[i*args.batch_size:(i+1)*args.batch_size, :] = (preds >= 0.5).cpu()
            allTargets[i*args.batch_size:(i+1)*args.batch_size, :] = targets.cpu()

            if args.log_freq and (i+1)%args.log_freq==0: logger.info(f"[{epoch+1}] ({(i+1)*args.batch_size}/{N} images) Loss = {totalTrainLoss/(i+1)}")
            totalTrainLoss += loss.item()/B
        
        # Training loss and best model checkpointing
        earlyStopCount += 1
        if totalTrainLoss < bestTrainLoss:
            bestTrainLoss = totalTrainLoss
            if args.early_stop_set == 'train': earlyStopCount = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': totalTrainLoss,
                }, os.path.join(args.o, 'checkpoints', 'best_train.pt'))

        # Training checkpoint
        if (epoch+1)%args.save_freq==0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': totalTrainLoss,
                }, os.path.join(args.o, 'checkpoints', 'latest.pt'))

        epochMetrics['train_loss'] = totalTrainLoss
        _addMetrics(allPreds, allTargets, 'train', epochMetrics)
        epochMetrics['train_time'] = time.time()-t

        if args.val_patients and (epoch+1)%args.val_freq==0:
            N = len(dataloaders['val'].dataset)
            B = len(dataloaders['val'])

            t = time.time()
            model.eval()
            totalValLoss = 0.0
            allPreds, allTargets = torch.zeros((N, 2)), torch.zeros((N, 2))

            with torch.no_grad():
                for i, data in enumerate(dataloaders['val']):
                    imgs, targets = data[0].to(device), data[1].to(device)
                    preds = torch.sigmoid(model(imgs))

                    bowelLoss = lossFn(preds[:, 0], targets[:, 0].to(torch.float))
                    extravLoss = lossFn(preds[:, 1], targets[:, 1].to(torch.float))
                    loss = bowelLoss + extravLoss

                    epochMetrics[f"val_loss_bowel"] = epochMetrics.get("val_loss_bowel", 0) + bowelLoss.item()/B
                    epochMetrics[f"val_loss_extrav"] = epochMetrics.get("val_loss_extrav", 0) + extravLoss.item()/B
                    allPreds[i*args.batch_size:(i+1)*args.batch_size, :] = (preds >= 0.5).cpu()
                    allTargets[i*args.batch_size:(i+1)*args.batch_size, :] = targets.cpu()

                    totalValLoss += loss.item()/B

            if totalValLoss < bestValLoss:
                bestValLoss = totalValLoss
                if args.early_stop_set == 'val': earlyStopCount = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': totalValLoss,
                    }, os.path.join(args.o, 'checkpoints', 'best_val.pt'))

        epochMetrics['val_loss'] = totalValLoss
        _addMetrics(allPreds, allTargets, 'val', epochMetrics)
        epochMetrics['val_time'] = time.time()-t
        
        print("\n---------------------------\n")
        for met, val in epochMetrics.items():
            logger.info(met + ": " + str(val))

        if not args.debug: wandb.log(epochMetrics)

        if args.early_stop_set != 'none' and earlyStopCount >= args.early_stop_epochs:
            logger.info("Stopping early") 
            return


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train segmentation model on images")
    datasetGroup = parser.add_argument_group("Dataset Info")
    datasetGroup.add_argument('--crop-h', type=int, default=512)
    datasetGroup.add_argument('--crop-w', type=int, default=512)
    datasetGroup.add_argument('--image-size', type=int, default=512)
    datasetGroup.add_argument('--train-patients')
    datasetGroup.add_argument('--val-patients')
    datasetGroup.add_argument('--num-workers', type=int, default=0)

    modelGroup = parser.add_argument_group("Model Parameters")
    modelGroup.add_argument('--arch', type=str, default='resnet50', choices=['resnet50'])
    #modelGroup.add_argument('--pretrained', action='store_true')

    trainGroup = parser.add_argument_group("Training Parameters")
    trainGroup.add_argument('--load', type=str, help="Path to training checkpoint to resume from") # TODO
    trainGroup.add_argument('--epochs', type=int, default=50, help="Maximum number of training epochs")
    trainGroup.add_argument('--val-freq', type=int, default=1, help="How often to evaluate validation set, if validation is enabled")
    trainGroup.add_argument('--early-stop-set', type=str, default='none', help="Which dataset to monitor loss on for early stopping. Can be 'train', 'val', or 'none'")
    trainGroup.add_argument('--early-stop-epochs', type=int, default=10, help="Number of epochs of no loss improvement before stopping")
    trainGroup.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    trainGroup.add_argument('--batch-size', type=int, default=32, help="Number of images per batch")

    outGroup = parser.add_argument_group("Output Info")
    outGroup.add_argument('-o', required=True, help="Output directory")
    outGroup.add_argument('--log-freq', type=int, default=100, help="Number of batches to log train loss")
    outGroup.add_argument('--save-freq', type=int, default=1, help="Number of epochs to save model checkpoint")
    outGroup.add_argument('--debug', action='store_true')
    
    args = parser.parse_args()
    logger.info(args)

    args = utils.dotdict(vars(args))
    utils.makedir(args.o)
    utils.makedir(os.path.join(args.o, 'checkpoints'))
    with open(os.path.join(args.o, 'config.json'), 'w') as f:
        json.dump(args, f) 

    if not args.debug: 
        os.environ['WANDB_MODE'] = "offline"
        wandb.init(project='rsna-atd', config=args, dir=args.o)

    start = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataloaders = {
        'train': dataset.getDataloader(args, 'train', patients=args.train_patients),
        'val': dataset.getDataloader(args, 'val', patients=args.val_patients) if args.val_patients else None
    }

    model = models.getModel(args).to(device)

    numParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Number of model parameters: " + str(numParams))
    train(args, model, dataloaders)
    logger.info(f"Total Training Time: {(time.time()-start):.0f} s")
