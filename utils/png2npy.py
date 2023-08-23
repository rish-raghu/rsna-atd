import os
import numpy as np
import cv2

CROP_SIZE=512

def centerCrop(img):
    startx = img.shape[1]//2-(CROP_SIZE//2)
    starty = img.shape[0]//2-(CROP_SIZE//2)    
    return img[starty:starty+CROP_SIZE,startx:startx+CROP_SIZE]

try: os.mkdir("data/train_npys")
except: pass

for patient in os.listdir("data/train_pngs"):
    try: os.mkdir(f"data/train_npys/{patient}")
    except: continue
    
    for series in os.listdir(f"data/train_pngs/{patient}"):
        pngs = os.listdir(f"data/train_pngs/{patient}/{series}")
        arr = np.zeros((len(pngs), CROP_SIZE, CROP_SIZE))
        for i, png in enumerate(pngs):
            img = cv2.imread(f"data/train_pngs/{patient}/{series}/{png}", cv2.IMREAD_GRAYSCALE)
            img = centerCrop(img)
            arr[i, ...] = img
        np.savez_compressed(f"data/train_npys/{patient}/{series}.npz", arr)
