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
from ast import literal_eval
import sys


path34 = "Tessting"
ash = 0
sap = 0
for i in range(0,5):
    inja = time.time()
    for image_path in os.listdir(path34):
        inja1 = time.time()
        if image_path == ".DS_Store":
            continue
        input_path1 = os.path.join(path34, image_path)
        print(input_path1)
        img = ReadImage.ReadImage1(input_path1)

        # json_str = json.dumps(img.tolist())
        # print(type(json.loads(json_str)),len(json.loads(json_str)),len(json.loads(json_str)[0]),type(np.array(json.loads(json_str))))
        sa = time.time()
        json_str = str(img.tolist())

        # print(type(eval(json_str)))
        print("hoshmand",time.time() - sa)
        # img = json.loads(json_str)
        print(time.time() - inja)
        sap += time.time() - inja1
        index.index("IMG",json_str)
        # out = subprocess.run('python tester.py ' + str(kam).replace(' ', ''), shell=True)

    ash += time.time() - inja
print((ash-sap)/75)


