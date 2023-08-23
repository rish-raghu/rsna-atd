import os
import pydicom
import time
import cv2
from PIL import Image
import numpy as np

# files = os.listdir("data/train_images/11633/41167")
# print(len(files))
# t = time.time()
# for file in files:
#     slice = pydicom.dcmread("data/train_images/11633/41167/" + file).pixel_array
#     #cv2.imwrite("data/train_images/pngs/" + file.replace('.dcm', '.png'), slice)
# print("DICOM time: ", time.time() - t)


# t = time.time()
# files = os.listdir("data/train_pngs/50385/55215/")
# arr = np.zeros((len(files), 512, 512))
# for i, file in enumerate(files):
#     arr[i, ...] = cv2.imread("data/train_pngs/26009/6135/" + file, cv2.IMREAD_GRAYSCALE)
# print("cv2 time: ", time.time() - t)


t = time.time()
arr = np.load("data/train_npys/50385/55215.npz")
print(arr['arr_0'].shape)
print("npy time: ", time.time() - t)


# t = time.time()
# for file in files:
#     slice = numpy.asarray(Image.open("data/train_pngs/26009/6135/" + file).convert('L'))
# print("PIL time: ", time.time() - t)

