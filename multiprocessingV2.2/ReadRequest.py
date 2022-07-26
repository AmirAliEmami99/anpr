import json
import os
import subprocess
import ConFindercopy
import cv2
import Partioningcopy
import numpy as np
import tensorflow as tf
import time
import ReadImage
import status
import GUI as G
import pickle
import index

path34 = "Tessting"
ash = 0
for i in range(0,5):
    inja = time.time()
    for image_path in os.listdir(path34):

        print(inja)
        if image_path == ".DS_Store":
            continue
        input_path1 = os.path.join(path34, image_path)
        print(input_path1)
        img = ReadImage.ReadImage1(input_path1)
        print(time.time() - inja)
        index.index("IMG",img)
    ash += time.time() - inja
print(ash/5)


