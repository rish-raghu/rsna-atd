# Source: https://www.kaggle.com/code/theoviel/get-started-quicker-dicom-png-conversion

import os
import pydicom
import cv2
import numpy as np

patients = os.listdir("data/train_images")
print("Patients: ", len(patients))
for i, patient in enumerate(patients):
    try: os.mkdir(f"/scratch/gpfs/ZHONGE/rraghu/train_pngs/{patient}")
    except: continue
    seriess = os.listdir(f"data/train_images/{patient}")
    for series in seriess:
        try: os.mkdir(f"/scratch/gpfs/ZHONGE/rraghu/train_pngs/{patient}/{series}")
        except: continue
        scans = os.listdir(f"data/train_images/{patient}/{series}")
        for scan in scans:
            dcm = pydicom.dcmread(f"data/train_images/{patient}/{series}/{scan}")
            pixel_array = dcm.pixel_array
            if dcm.PixelRepresentation == 1:
                bit_shift = dcm.BitsAllocated - dcm.BitsStored
                dtype = pixel_array.dtype 
                pixel_array = (pixel_array << bit_shift).astype(dtype) >>  bit_shift
            
            intercept = float(dcm.RescaleIntercept)
            slope = float(dcm.RescaleSlope)
            center = int(dcm.WindowCenter)
            width = int(dcm.WindowWidth)
            low = center - width / 2
            high = center + width / 2    
            pixel_array = (pixel_array * slope) + intercept
            pixel_array = np.clip(pixel_array, low, high)

            pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-6)
            
            if dcm.PhotometricInterpretation == "MONOCHROME1":
                pixel_array = 1 - pixel_array
            
            cv2.imwrite(f"/scratch/gpfs/ZHONGE/rraghu/train_pngs/{patient}/{series}/{scan.replace('.dcm', '.png')}", (pixel_array * 255).astype(np.uint8))
    
    print(f"{i+1} done, patient {patient}", flush=True)
