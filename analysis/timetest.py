import os
import pydicom
import time
import cv2
from PIL import Image
import numpy

# files = os.listdir("data/train_images/11633/41167")
# print(len(files))
# t = time.time()
# for file in files:
#     slice = pydicom.dcmread("data/train_images/11633/41167/" + file).pixel_array
#     #cv2.imwrite("data/train_images/pngs/" + file.replace('.dcm', '.png'), slice)
# print("DICOM time: ", time.time() - t)

files = os.listdir("data/train_pngs/26009/6135/")


# t = time.time()
# for file in files:
#     slice = cv2.imread("data/train_pngs/26009/6135/" + file, cv2.IMREAD_GRAYSCALE)
# print("cv2 time: ", time.time() - t)

t = time.time()
for file in files:
    slice = numpy.asarray(Image.open("data/train_pngs/26009/6135/" + file).convert('L'))
print("PIL time: ", time.time() - t)

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
