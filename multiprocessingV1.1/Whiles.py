from multiprocessing import Process , Manager
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





def ReadImages(HaveRead, Haveconfinded, HaveConPlate, HaveNumber):
    # global HaveRead,Haveconfinded,HaveConPlate,HaveNumber
    path34 = "/Users/amirsajjad/Desktop/CarID_OCR/multiprocessing/Tessting"

    for image_path in os.listdir(path34):
        if image_path == ".DS_Store":
            continue
        input_path1 = os.path.join(path34, image_path)
        img = ReadImage.ReadImage1(input_path1)
        HaveRead.append(img)
        print(len(HaveRead),input_path1)
def contoro(HaveRead, Haveconfinded, HaveConPlate, HaveNumber):
    # global HaveRead,Haveconfinded,HaveConPlate,HaveNumber
    while True:
        # try:
        if len(HaveRead) > 0:
            sa = time.time()
            sham = ConFindercopy.protot(HaveRead.pop(0))
            print("------------------------------>>>>>Time",time.time() - sa)
            Haveconfinded.append(sham)
        # except:
        #     pass
def PlateorNot(HaveRead, Haveconfinded, HaveConPlate, HaveNumber):
    # global HaveRead,Haveconfinded,HaveConPlate,HaveNumber
    while True:
        # try:
        if len(Haveconfinded) > 0:
            sham = PlateOrNot.PorNP(Haveconfinded.pop(0))
            HaveConPlate.append(sham)
        # except:
        #     pass
def Platenumbers(HaveRead, Haveconfinded, HaveConPlate, HaveNumber):
    # global HaveRead,Haveconfinded,HaveConPlate,HaveNumber
    while True:
        # try:
            # print(len(HaveConPlate))
        if len(HaveConPlate) > 0:
            sham = Partioningcopy.finncc(HaveConPlate.pop(0))
            HaveNumber.append(sham)
            print(len(HaveNumber),"Time",time.time())

        # except:
        #     pass
def init1(HaveRead, Haveconfinded, HaveConPlate, HaveNumber):
    # global HaveRead,Haveconfinded,HaveConPlate,HaveNumber
    while True:
        print("HaveRead",len(HaveRead),"Haveconfinded",len(Haveconfinded),"HaveConPlate",len(HaveConPlate),"HaveNumber",len(HaveNumber))



# HaveRead, Haveconfinded, HaveConPlate, HaveNumber = [], [], [], []

if __name__ == '__main__':
    print("Start location")
    with Manager() as manager:
        HaveRead = manager.list([])
        Haveconfinded = manager.list([])
        HaveConPlate = manager.list([])
        HaveNumber = manager.list([])
        RI = Process(target=ReadImages, args=(HaveRead, Haveconfinded, HaveConPlate, HaveNumber))
        Co = Process(target=contoro, args=(HaveRead, Haveconfinded, HaveConPlate, HaveNumber))
        PNP = Process(target=PlateorNot, args=(HaveRead, Haveconfinded, HaveConPlate, HaveNumber))
        Num = Process(target=Platenumbers, args=(HaveRead, Haveconfinded, HaveConPlate, HaveNumber))
        stat = Process(target=init1, args=(HaveRead, Haveconfinded, HaveConPlate, HaveNumber))
        print("Start location1")
        RI.start()
        RI.join()
        Co.start()
        PNP.start()
        Num.start()
        # stat.start()
        Co.join()
        PNP.join()
        Num.join()
        # stat.join()