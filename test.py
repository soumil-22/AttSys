import cv2
import os
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

headshots_folder_name = 'C:\\Users\\soumil\\Desktop\\Python Projects\\AttSys\\Headshots'
for root, _, files in os.walk(headshots_folder_name):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
            # path of the image
            path = os.path.join(root, file)
            #path = file
            print(path)
            img = cv2.imread(path)
            cv2.imshow(file, img)
            #print(root, _, file)