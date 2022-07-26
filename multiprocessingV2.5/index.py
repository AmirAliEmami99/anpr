from multiprocessing import Process , Manager,Queue
import multiprocessing as mp
from threading import Thread
import json
import os
import subprocess
import cv2
import numpy as np
import tensorflow as tf
import time
# import status
import ConFindercopy
import pickle
import ReadImage
import PlateOrNot
import Partioningcopy

def index(jens,data,mod1=0,mod2=0,mod3=0):
    if jens == "IMG":
        pass
        ConFindercopy.protot(data,mod1,mod2,mod3)
    elif jens == "CON":
        pass
        PlateOrNot.PorNP(data,mod1,mod2,mod3)
    elif jens == "TrustedCON":
        pass
        try:
            Partioningcopy.finncc(data,mod1,mod2,mod3)
        except:
            pass
    elif jens == "Number":
        pass
        # print("tahesh",time.time())
        # print("pelak",data)
