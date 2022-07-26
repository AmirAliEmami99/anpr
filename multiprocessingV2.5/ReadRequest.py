# import json
import os
# import subprocess
# import ConFindercopy
# import cv2
# import Partioningcopy
# import numpy as np
import tensorflow as tf
import time
import ReadImage
# import status
# import GUI as G
# import pickle
import sys
import index

# def masaale(ki):
#
#     new_model1 = tf.keras.models.load_model("zeroToNineOurDataset.h5")
#     aplphabet = tf.keras.models.load_model('horof.h5')
#     new_model = tf.keras.models.load_model('pNCompleteDatasetModel.h5')
#     print(time.time())
#     path34 = "Tessting"
#
#     image_path = os.listdir(path34)[ki]
#     if image_path != ".DS_Store":
#         input_path1 = os.path.join(path34, image_path)
#         # print(input_path1)
#         img = ReadImage.ReadImage1(input_path1)
#         index.index("IMG",img,new_model,new_model1,aplphabet)
#
# if __name__ == "__main__":
#     sham = sys.argv[1]
#     # print("Hello, World!",sham)
#     masaale(int(sham))


def masaale(ki):

    new_model1 = tf.keras.models.load_model("zeroToNineOurDataset.h5")
    aplphabet = tf.keras.models.load_model('horof.h5')
    new_model = tf.keras.models.load_model('pNCompleteDatasetModel.h5')
    print("shoroe",time.time())
    kaj = 0
    kol = 0
    path34 = "Tessting"
    es = 0
    # print(len(os.listdir(path34)))
    for i in range((ki-1)*45,ki*45+1):
        image_path = os.listdir(path34)[i]
        if image_path == ".DS_Store":
            continue
        es += 1
        mish = time.time()
        input_path1 = os.path.join(path34, image_path)
        # print("Address",input_path1)
        img = ReadImage.ReadImage1(input_path1)
        kaj += time.time() - mish
        mish1 = time.time()
        index.index("IMG",img,new_model,new_model1,aplphabet)
        kol += time.time() - mish1
    print(kaj/es,kol/es,time.time(),es)
if __name__ == "__main__":
    sham = sys.argv[1]
    # print("Hello, World!",sham)
    masaale(int(sham))

