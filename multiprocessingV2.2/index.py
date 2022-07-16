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

def index(jens,data):
    if jens == "IMG":
        pass
        ConFindercopy.protot(data)
    elif jens == "CON":
        pass
        PlateOrNot.PorNP(data)
    elif jens == "TrustedCON":
        pass
        Partioningcopy.finncc(data)
    elif jens == "Number":
        print(time.time())
        print("pelak",data)
