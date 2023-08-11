import os
import argparse
import logging
import time

import torch
from torch import nn
import numpy as np

import models
import dataset
import utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')


def train(args, model, dataloaders):
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
    lossFn = nn.CrossEntropyLoss()
    bestTrainLoss = float('inf')
    bestValLoss = float('inf')
    earlyStopCount = 0

    for epoch in range(args.epochs):
        totalTrainLoss = 0.0
        t = time.time()
        model.train()
        
        for i, data in enumerate(dataloaders['train']):
            imgs, targets = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            bowelLoss = lossFn(preds[0], targets[:, 0].to(device))
            extravLoss = lossFn(preds[1], targets[:, 1].to(device))
            kidneyLoss = lossFn(preds[2], targets[:, 2].to(device))
            liverLoss = lossFn(preds[3], targets[:, 3].to(device))
            spleenLoss = lossFn(preds[4], targets[:, 4].to(device))
            loss = bowelLoss + extravLoss + kidneyLoss + liverLoss + spleenLoss
            loss.backward()
            optimizer.step()
            
            logger.info("[{}] ".format(epoch+1) + "Batch loss = " + str(loss.item()))
            totalTrainLoss += loss.item()
        
        # Training loss and best model checkpointing
        earlyStopCount += 1
        if totalTrainLoss < bestTrainLoss:
            bestTrainLoss = totalTrainLoss
            if args.early_stop_set == 'train': earlyStopCount = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, os.path.join(args.o, 'checkpoints', 'best_train.pt'))

        # Training checkpoint
        if (epoch+1)%args.save_freq==0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, os.path.join(args.o, 'checkpoints', 'latest.pt'))

        logger.info(f"[{epoch + 1}] Epoch {epoch+1} | Train loss: {totalTrainLoss / len(dataloaders['train'].dataset):.5f} | Time: {(time.time()-t):.1f} s")

        if args.val and (epoch+1)%args.val_freq==0:
            t = time.time()
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for i, data in enumerate(dataloaders['val']):
                    imgs = data[0].to(device)
                    targets = torch.squeeze(data[1].to(device).long())
                    preds = model(imgs)['out']
                    total_val_loss += lossFn(preds, targets).item()
                if args.vis_val:
                    visualize(imgs, preds, os.path.join(args.o, 'val_samples', 'seg_' + str(epoch+1)))

            if total_val_loss < bestValLoss:
                bestValLoss = total_val_loss
                if args.early_stop_set == 'val': earlyStopCount = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, os.path.join(args.o, 'checkpoints', 'best_val.pt'))
            logger.info(f"[{epoch + 1}] Epoch {epoch+1} | Validation loss: {total_val_loss / dataloaders['valSize']:.5f} | Time: {(time.time()-t):.1f} s")
        
        if args.early_stop_set != 'none' and earlyStopCount >= args.early_stop_epochs:
            logger.info("Stopping early") 
            return


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train segmentation model on images")
    datasetGroup = parser.add_argument_group("Dataset Info")
    #datasetGroup.add_argument('datadir', type=str, help='Directory with images and ground truth segmentations')
    datasetGroup.add_argument('--val', action='store_true', help="Whether to evaluate on a validation set")

    modelGroup = parser.add_argument_group("Model Parameters")
    modelGroup.add_argument('--arch', type=str, default='unet2d', help='unet2d, deeplabv3 (resnet50)')
    #modelGroup.add_argument('--depth', type=int, default=5, help="Number of blocks in reducing path")

    trainGroup = parser.add_argument_group("Training Parameters")
    # TODO
    trainGroup.add_argument('--load', type=str, help="Path to training checkpoint to resume from")
    trainGroup.add_argument('--epochs', type=int, default=50, help="Maximum number of training epochs")
    trainGroup.add_argument('--val-freq', type=int, default=1, help="How often to evaluate validation set, if validation is enabled")
    trainGroup.add_argument('--early-stop-set', type=str, default='none', help="Which dataset to monitor loss on for early stopping. Can be 'train', 'val', or 'none'")
    trainGroup.add_argument('--early-stop-epochs', type=int, default=5, help="Number of epochs of no loss improvement before stopping")
    trainGroup.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    trainGroup.add_argument('--batch-size', type=int, default=8, help="Number of images per batch")

    outGroup = parser.add_argument_group("Output Info")
    outGroup = parser.add_argument('-o', required=True, help="Output directory")
    # TODO
    outGroup = parser.add_argument('--log-freq', type=int, default=1000, help="Number of iterations to log train_loss")
    outGroup = parser.add_argument('--save-freq', type=int, default=1, help="Number of epochs to save model checkpoint")
    # outGroup = parser.add_argument('--vis-train', action='store_true', help="Save predictions on a training subset each epoch")
    # outGroup = parser.add_argument('--vis-val', action='store_true', help="Save predictions on a validation subset each time it is evaluated, if validation is enabled")
    args = parser.parse_args()
    logger.info(args)

    start = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    utils.makedir(args.o)
    utils.makedir(os.path.join(args.o, 'checkpoints')) 
    # if args.vis_train: utils.makedir(os.path.join(args.o, 'train_samples')) 
    # if args.vis_val: utils.makedir(os.path.join(args.o, 'val_samples')) 
    
    dataloaders = {'train': dataset.getDataloader(args.batch_size)}
    model = models.UNet2D(32).to(device)
    
    # Initialize lazy layers
    dummyBatch = torch.ones((8, 32, 256, 256)).to(device)
    model(dummyBatch)
    
    numParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Number of model parameters: " + str(numParams))
    train(args, model, dataloaders)
    logger.info(f"Total Time: {(time.time()-start):.0f} s")
