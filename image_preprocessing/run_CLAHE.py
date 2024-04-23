import os
import cv2
import numpy as np

def clahe(img, clip_limit=4, tile_grid_size=(8, 8)):
    if img.dtype != np.uint8:
        raise TypeError("clahe supports only uint8 inputs")

    clahe_mat = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(img.shape) == 2 or img.shape[2] == 1:
        img = clahe_mat.apply(img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        img[:, :, 0] = clahe_mat.apply(img[:, :, 0])
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    return img

yourPath = "./test/"
allFileList = os.listdir(yourPath)
for file in allFileList:
    if os.path.isfile(yourPath+file):
        image = cv2.imread(yourPath+file)
        image_augmented = clahe(img = image)
        cv2.imwrite("./test/result"+"/"+file, image_augmented,[cv2.IMWRITE_PNG_COMPRESSION, 0])