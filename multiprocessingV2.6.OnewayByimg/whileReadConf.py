import json
import os
import subprocess
import cv2
import numpy as np
import tensorflow as tf
import time
import status
import ConFindercopy
import pickle

while True:
    try:
        with open("HaveRead.pk", 'rb') as fi:
            HaveRead = pickle.load(fi)
            t01 = time.time()
            fi.close()
        if len(HaveRead) > 0:
            print("avalie",time.time())
            with open("HaveRead.pk", 'w+b') as fi:
                sham = ConFindercopy.protot(HaveRead.pop(0))
                pickle.dump(HaveRead, fi)
            fi.close()
            with open("Haveconfinded.pk", 'r+b') as fi:
                Haveconfinded = pickle.load(fi)
                fi.close()
            with open("Haveconfinded.pk", 'w+b') as fi:
                Haveconfinded.append(sham)
                # print(Haveconfinded)
                pickle.dump(Haveconfinded, fi)
                fi.close()
            print(time.time() - t01)
    except:
        pass



