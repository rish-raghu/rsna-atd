import os
import argparse
import logging
import time

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import wandb

import models
import dataset
import utils
import metrics

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

def __addAccuracies(split, epochMetrics, organ, labelAccs):
    epochMetrics[f"{split}_acc_{organ}_healthy"] = epochMetrics.get(f"{split}_acc_{organ}_healthy", 0) + labelAccs[0].item()
    if len(labelAccs)==3:
        epochMetrics[f"{split}_acc_{organ}_low"] = epochMetrics.get(f"{split}_acc_{organ}_low", 0) + labelAccs[1].item()
        epochMetrics[f"{split}_acc_{organ}_high"] = epochMetrics.get(f"{split}_acc_{organ}_high", 0) + labelAccs[2].item()
    else:
        epochMetrics[f"{split}_acc_{organ}_injured"] = epochMetrics.get(f"{split}_acc_{organ}_injured", 0) + labelAccs[1].item()


def train(args, model, dataloaders):
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
    lossFns = {
        'bowel': nn.NLLLoss(weight=torch.Tensor([1, 2]).to(device)),
        'extravasation': nn.NLLLoss(weight=torch.Tensor([1, 6]).to(device)),
        'liver': nn.NLLLoss(weight=torch.Tensor([1, 2, 4]).to(device)),
        'kidney': nn.NLLLoss(weight=torch.Tensor([1, 2, 4]).to(device)),
        'spleen': nn.NLLLoss(weight=torch.Tensor([1, 2, 4]).to(device))
    }
    bestTrainLoss = float('inf')
    bestValLoss = float('inf')
    earlyStopCount = 0

    for epoch in range(args.epochs):
        totalTrainLoss = 0.0
        epochMetrics = {}
        t = time.time()
        model.train()
        
        for i, data in enumerate(dataloaders['train']):
            imgs, targets = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = 0
            for j, organ in enumerate(lossFns.keys()):
                organPreds, organTargets = preds[j], targets[:, j].to(device)
                organLoss = lossFns[organ](F.log_softmax(organPreds, dim=1), organTargets)
                epochMetrics[f"train_loss_{organ}"] = epochMetrics.get(f"train_loss_{organ}", 0) + organLoss.item()
                loss += organLoss
                labelAccs = metrics.label_wise_accuracy(organPreds, organTargets)
                __addAccuracies('train', epochMetrics, organ, labelAccs)
            loss.backward()
            optimizer.step()
            
            if epoch==0: logger.info("[{}] ".format(epoch+1) + "Batch loss = " + str(loss.item()))
            totalTrainLoss += loss.item()
            if i==2: break
        
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

        for met in epochMetrics.keys():
            epochMetrics[met] /= len(dataloaders['train'].dataset)
        epochMetrics['train_loss'] = totalTrainLoss / len(dataloaders['train'].dataset)
        epochMetrics['train_time'] = time.time()-t
        #logger.info(f"[{epoch + 1}] Epoch {epoch+1} | Train loss: {epochMetrics['train_loss']:.5f} | Time: {(epochMetrics['train_time']):.1f} s")

        if args.val_patients and (epoch+1)%args.val_freq==0:
            t = time.time()
            model.eval()
            totalValLoss = 0.0
            with torch.no_grad():
                for i, data in enumerate(dataloaders['val']):
                    imgs, targets = data[0].to(device), data[1].to(device)
                    preds = model(imgs)
                    losses = {}
                    for j, organ in enumerate(lossFns.keys()):
                        organPreds, organTargets = preds[j], targets[:, j].to(device)
                        organLoss = lossFns[organ](F.log_softmax(organPreds, dim=1), organTargets)
                        epochMetrics[f"val_loss_{organ}"] = epochMetrics.get(f"val_loss_{organ}", 0) + organLoss.item()
                        loss += organLoss
                        labelAccs = metrics.label_wise_accuracy(organPreds, organTargets)
                        __addAccuracies('val', epochMetrics, organ, labelAccs)
                    totalValLoss += loss.item()
                    if i==2: break

            if totalValLoss < bestValLoss:
                bestValLoss = totalValLoss
                if args.early_stop_set == 'val': earlyStopCount = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': totalValLoss,
                    }, os.path.join(args.o, 'checkpoints', 'best_val.pt'))

            for met in epochMetrics.keys():
                if met.startswith('val'): epochMetrics[met] /= len(dataloaders['val'].dataset)
            epochMetrics['val_loss'] = totalValLoss / len(dataloaders['val'].dataset)
            epochMetrics['val_time'] = time.time()-t
            #logger.info(f"[{epoch + 1}] Epoch {epoch+1} | Validation loss: {epochMetrics['val_loss']:.5f} | Time: {epochMetrics['val_time']:.1f} s")
        
        for met, val in epochMetrics.items():
            logger.info(met + ": " + str(val))

        if not args.debug: wandb.log(epochMetrics)

        if args.early_stop_set != 'none' and earlyStopCount >= args.early_stop_epochs:
            logger.info("Stopping early") 
            return


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train segmentation model on images")
    datasetGroup = parser.add_argument_group("Dataset Info")
    datasetGroup.add_argument('--dataset-type', default='volume', help='volume, slice_sampler')
    datasetGroup.add_argument('--image-size', type=int, default=512)
    datasetGroup.add_argument('--z-size', type=int, default=32, help='Number of slices for a slice_sampler, or z-dimension for downsampled volume')
    datasetGroup.add_argument('--train-patients')
    datasetGroup.add_argument('--val-patients')
    datasetGroup.add_argument('--num-workers', type=int, default=1)

    modelGroup = parser.add_argument_group("Model Parameters")
    modelGroup.add_argument('--arch', type=str, default='unet2d', help='unet2d, unet3d')
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
    outGroup.add_argument('-o', required=True, help="Output directory")
    # TODO
    outGroup.add_argument('--log-freq', type=int, default=1000, help="Number of iterations to log train_loss")
    outGroup.add_argument('--save-freq', type=int, default=1, help="Number of epochs to save model checkpoint")
    outGroup.add_argument('--debug', action='store_true')
    
    args = parser.parse_args()
    logger.info(args)

    utils.makedir(args.o)
    utils.makedir(os.path.join(args.o, 'checkpoints')) 

    if not args.debug: 
        os.environ['WANDB_MODE'] = "offline"
        wandb.init(project='rsna-atd', config=args, dir=args.o)

    start = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataloaders = {
        'train': dataset.getDataloader(args, 'train'),
        'val': dataset.getDataloader(args, 'val') if args.val_patients else None
    }

    

    model = models.get_model(args).to(device)
    
    # Initialize lazy layers
    dummyBatch = torch.ones((args.batch_size, args.z_size, args.image_size, args.image_size)).to(device)
    model(dummyBatch)

    # iterator = iter(dataloaders['train'])
    # t1 = time.time()
    # batch1 = next(iterator)
    # print("Batch 1 time ", time.time() - t1)
    # t1 = time.time()
    # model(batch1[0].to(device))
    # print("Inf 1 time ", time.time() - t1)
    # t1 = time.time()
    # batch2 = next(iterator)
    # print("Batch 2 time ", time.time() - t1)
    # t1 = time.time()
    # model(batch2[0].to(device))
    # print("Inf 2 time ", time.time() - t1)
    # assert False

    # t1 = time.time()
    # batch1 = dataloaders['train'].dataset[0]
    # print("Batch 1 time ", time.time() - t1)
    # t1 = time.time()
    # batch1 = dataloaders['train'].dataset[0]
    # print("Batch 1 time ", time.time() - t1)
    # assert False

    numParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Number of model parameters: " + str(numParams))
    train(args, model, dataloaders)
    logger.info(f"Total Time: {(time.time()-start):.0f} s")
