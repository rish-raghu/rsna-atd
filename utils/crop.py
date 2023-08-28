import cv2
import numpy as np
import os

for file in os.listdir("."):
    if file.endswith(".png") and 'cropped' not in file:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #img = cv2.drawContours(img, contours, -1, (255, 0, 0), 3)
        areas = [cv2.contourArea(contour) for contour in contours]
        largest_contour = contours[np.argmax(areas)]
        #img = cv2.drawContours(img, [largest_contour], 0, (255, 0, 0), 3)
        mask = cv2.drawContours(np.zeros_like(img), [largest_contour], 0, 255, cv2.FILLED)
        img = cv2.bitwise_and(img, mask)
        xmin, xmax = np.min(largest_contour[:, 0, 0]), np.max(largest_contour[:, 0, 0])
        ymin, ymax = np.min(largest_contour[:, 0, 1]), np.max(largest_contour[:, 0, 1])
        img = img[ymin:ymax+1, xmin:xmax+1]
        cv2.imwrite(file.split('.png')[0] + "_cropped.png", img)
