import json
import os
import subprocess
import ConFindercopy
import cv2
import Partioningcopy
import numpy as np
import tensorflow as tf
import time
import status
import Partioningcopy
import pickle

while True:
    try:
        with open("HaveConPlate.pk", 'rb') as fi:
            HaveConPlate = pickle.load(fi)
            fi.close()

        if len(HaveConPlate) > 0:
            with open("HaveConPlate.pk", 'w+b') as fi:
                sham = Partioningcopy.finncc(HaveConPlate.pop(0))
                pickle.dump(HaveConPlate, fi)
            fi.close()

            with open("HaveNumber.pk", 'r+b') as fi:
                HaveNumber = pickle.load(fi)
                fi.close()

            with open("HaveNumber.pk", 'w+b') as fi:
                HaveNumber.append(sham)
                print(HaveNumber)
                pickle.dump(HaveNumber, fi)
                fi.close()
            print("avalie",time.time())
    except:
        pass



