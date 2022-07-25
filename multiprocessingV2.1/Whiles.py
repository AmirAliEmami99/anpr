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





def ReadImages(HaveRead):
    # global HaveRead,Haveconfinded,HaveConPlate,HaveNumber
    path34 = "/Users/amirsajjad/Desktop/CarID_OCR/multiprocessingV2.1/Tessting"

    for image_path in os.listdir(path34):
        if image_path == ".DS_Store":
            continue
        input_path1 = os.path.join(path34, image_path)
        img = ReadImage.ReadImage1(input_path1)
        HaveRead.append(img)
        print(len(HaveRead),input_path1)
    return HaveRead
def contoro(HaveRead, Haveconfinded, HaveConPlate, HaveNumber):

    #     sa = time.time()
    sham = ConFindercopy.protot(HaveRead.pop(0))
        # print("------------------------------>>>>>Time",time.time() - sa)
    Haveconfinded.append(sham)

def PlateorNot(HaveRead, Haveconfinded, HaveConPlate, HaveNumber):

    sham = PlateOrNot.PorNP(Haveconfinded.pop(0))
    HaveConPlate.append(sham)

def Platenumbers(HaveRead, Haveconfinded, HaveConPlate, HaveNumber):
    sham = Partioningcopy.finncc(HaveConPlate.pop(0))
    HaveNumber.append(sham)

def init1(HaveRead, Haveconfinded, HaveConPlate, HaveNumber):
    # global HaveRead,Haveconfinded,HaveConPlate,HaveNumber
    while True:
        print("HaveRead",len(HaveRead),"Haveconfinded",len(Haveconfinded),"HaveConPlate",len(HaveConPlate),"HaveNumber",len(HaveNumber))



# HaveRead, Haveconfinded, HaveConPlate, HaveNumber = [], [], [], []

if __name__ == '__main__':
    print("Start location")
    HaveRead = []
    HaveRead = ReadImages(HaveRead)
    mp.set_start_method('spawn')
    print("Start location")

    with Manager() as manager:
        HaveRead = manager.list(HaveRead)
        Haveconfinded = manager.list([])
        HaveConPlate = manager.list([])
        HaveNumber = manager.list([])
        while True:
            try:
                print("HaveRead", len(HaveRead), "Haveconfinded", len(Haveconfinded), "HaveConPlate", len(HaveConPlate),"HaveNumber", len(HaveNumber))

                p1.join()
                p2.join()
                p3.join()
                # Haveconfinded.append(q1)
                # HaveConPlate.append(q2)
                # HaveNumber.append(q3)
            except:
                print("inja")
                pass
            print("inja1111")
            if len(HaveRead) > 0 :
                p1 = Process(target=contoro,args=(HaveRead, Haveconfinded, HaveConPlate, HaveNumber))
                p1.start()
            if len(Haveconfinded) > 0:

                p2 = Process(target=PlateorNot,args=(HaveRead, Haveconfinded, HaveConPlate, HaveNumber))
                p2.start()
            if len(HaveConPlate) > 0:
                p3 = Process(target=Platenumbers,args=(HaveRead, Haveconfinded, HaveConPlate, HaveNumber))
                p3.start()
            print(time.time())
