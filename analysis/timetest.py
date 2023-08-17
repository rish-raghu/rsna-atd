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
