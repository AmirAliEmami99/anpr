import json
import os
import subprocess
import ConFindercopy
import cv2
import Partioningcopy
import numpy as np
import tensorflow as tf
import time
import PlateOrNot
import pickle

while True:
    try:

        with open("Haveconfinded.pk", 'rb') as fi:
            Haveconfinded = pickle.load(fi)
            fi.close()

        if len(Haveconfinded) > 0:
            with open("Haveconfinded.pk", 'w+b') as fi:
                sham = PlateOrNot.PorNP(Haveconfinded.pop(0))
                pickle.dump(Haveconfinded, fi)
                fi.close()

            with open("HaveConPlate.pk", 'r+b') as fi:
                HaveConPlate = pickle.load(fi)
                fi.close()

            with open("HaveConPlate.pk", 'w+b') as fi:
                HaveConPlate.append(sham)
                print(HaveConPlate)
                pickle.dump(HaveConPlate, fi)
                fi.close()
    except:
        pass

