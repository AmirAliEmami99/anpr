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

# new_model1 = tf.keras.models.load_model('zeroToNineOurDataset.h5')


# #2- call the Pelak finder
# def FindContures(img):
#     return ConFindercopy.protot(img,12, 0, 0)


path34 = "/Users/amirsajjad/Desktop/CarID_OCR/multiiprocessing/Tessting"

for image_path in os.listdir(path34):
    if image_path == ".DS_Store":
        continue
    input_path1 = os.path.join(path34, image_path)
    print(input_path1)
    img = ReadImage.ReadImage1(input_path1)
    halle = 1
    while halle:
        try:
            with open("HaveRead.pk", 'r+b') as fi:
                HaveRead = pickle.load(fi)
            fi.close()
            with open("HaveRead.pk", 'w+b') as fi:
                HaveRead.append(img)
                pickle.dump(HaveRead, fi)
            # status.HaveRead.append(img)
            fi.close()
            halle = 0
        except:
            pass
    # print("Reading Time :",t11 - t01)
    # contures = FindContures(img)
    # t21 = time.time()
    # lastres = PorNP(contures)
    # t3 = time.time()
    # pathname = makeDIR("/Users/amirsajjad/Desktop/CarID_OCR/Pelaks", input_path1, lastres, contures,img)
    # t31 = time.time()
    # hamin = []
    # for i, j, z in zip(contures, lastres, pathname):
    #     print(j)
    #     if j < 0.3:
    #         # t30 = time.time()
    #         lists = i.tolist()
    #         json_str = json.dumps(lists)
    #
    #         # chars = Partioningcopy.finncc(json_str)
    #         # print("shahid",chars)
    #         # imgsummery.append([i,chars])
    #         # hamin.append(chars)
    #         # subprocess.call(['python3.9', 'T1.py'])
    #         # print(i)
    #         # print(z)
    #         # ""
    #         emam = subprocess.run("source /Users/amirsajjad/opt/miniconda3/etc/profile.d/conda.sh;conda activate myenv36; python3 Partioningcopy.py "+json_str, shell=True, capture_output=True)
    #         print("saa",emam)
    #
    #         # print(subprocess.call(['python3.9', 'Partioningcopy.py', json_str]))
    #
    #         # print(subprocess.call(['python3.9', 'Partioningcopy.py', json_str, new_model1]))
    #
    #
    #         # print("predicting :", time.time() - t30)
    #     # print(z, j)
    # # G.Page(image_path,len(imgsummery),hamin)
    # print("imgsummm",imgsummery)
    # t4 = time.time()
    #
    # print("Finding Contures Time :",t21 - t11)
    # print("Pelak not pelak Time :", t3 - t21)
    # print("SAving time:", t4 - t31 )
    # print("DarKol",t21 - t11 + t3 - t21 + t4 - t31 )

    # kol = []
    # t30 = time.time()
    # for i, j, z in zip(contures, lastres, pathname):
    #     if j < 0.3:
    #         kol.append(contures)
    #     # print(z, j)
    #
    # print("predicting :", time.time() - t30)
    # print(kol)
    # chars = Partioningcopy.finncc1(kol, pathname, new_model1)
    # print("hjhj",chars)

    # except Exception as e:
    #     print('Failed to upload to ftp: ' + str(e))
    #     pass
