# from multiprocessing import Pool


import sys
# path location to find CV2
import cv2


# from matplotlib import pyplot as plt
import numpy as np
# import imutils
import time
import os
import copy
import pandas as pd
import tensorflow as tf
#import defs
import json
import index

new_model1 = tf.keras.models.load_model("zeroToNineOurDataset.h5")
aplphabet = tf.keras.models.load_model('horof.h5')


horoooof = ["Alef", "Be", "Pe", "Te", "Se3Noghte", "jim", "Dal", "Zhe","Sin", "She", "Sad", "TeDasteDar", "Ein", "Phe", "ghaf", "Lam", "mim","Non", "Vav", "He", "ie"]

# def finncc(img,new_model2=0,where=0,image_path = 0):
#     # print(tf)
#     img = json.loads(img[0])
#     tt1 = time.time()
#     min1 = np.min(img)
#     max1 = np.max(img)
#     img = ((img - min1) / (max1 - min1)) * 255
#     #
#     # plt.imshow(img.astype('uint8'),cmap='gray', vmin=0, vmax=255)
#     # plt.show()
#
#     flag1 = 0
#     vahid = ""
#     # print("-0-0-0-0 ",img.shape)
#     # print(np.max(img))
#     t01 = time.time()
#     for tol in range(0,img.shape[1]):
#         if flag1:
#             break
#         for s in range(0,img.shape[0]):
#             if img[s, tol] > 90:
#                 sahand = tol
#                 flag1 = 1
#                 break
#     flag1 = 0
#     for tol in range(img.shape[1]-1,0,-1):
#         if flag1:
#             break
#         for s in range(0,int(img.shape[0]*0.3)):
#             if img[s, tol] > 90:
#                 lastsahand = tol
#                 flag1 = 1
#                 # print(lastsahand)
#                 break
#         aya = 0
#         if flag1 ==1:
#             for s1 in range(img.shape[0]-1, int(img.shape[0] * 0.8),-1):
#                 # print(img[s1, tol],s1)
#                 if img[s1, tol] > 90:
#                     lastsahand = tol
#                     aya = 1
#                     break
#         if aya == 0:
#             flag1 = 0
#     # print("===>",sahand,lastsahand)
#     t2 = time.time() - t01
#
#     img = cv2.resize(img[:,sahand:lastsahand], (250, 60))
#
#     tt2 = time.time()
#     # print("first phase",tt2-tt1)
#     # print(np.max(img))
#
#     # plt.imshow(img.astype('uint8'),cmap='gray', vmin=0, vmax=255)
#     # plt.show()
#
#     counter = 0
#     prediictlist = []
#
#     for i in range(0+22,img.shape[1],25):
#         t01 = time.time()
#         if counter >= 9:
#             break
#         tdakheli = time.time()
#         j = i + 25
#         minbala = 0
#         minpaeen = 500
#         for t in range(0, img.shape[0]):
#             if img[t, i] > 50:
#                 flag = 1
#                 minbala = t
#                 break
#         for t in range(img.shape[0] - 1, 0, -1):
#             if img[t, i] > 50:
#                 flag = 1
#                 minpaeen = t
#                 break
#         t2 += time.time() - t01
#         #print("---------------------------")
#         # tvasat = time.time() - tdakheli
#         if counter in [0,1,4,5,6,7,8]:
#             if counter > 6 :
#                 i = i + 3
#                 j = j + 3
#                 min1 = np.min(img[minbala :minpaeen , i:j])
#                 max1 = np.max(img[minbala :minpaeen, i:j])
#
#                 min2 = np.min(img[:, i:j])
#                 max2 = np.max(img[:, i:j])
#
#                 # img[minbala + 10:minpaeen-2, i:j] = ((img[minbala + 10:minpaeen-2, i:j] - min1) / (max1 - min1)) * 255
#                 sal = ((img[minbala:minpaeen, i:j] - min1) / (max1 - min1)) * 255
#                 sal2 = ((img[:, i:j] - min2) / (max2 - min2)) * 255
#
#                 felan12 = cv2.resize(sal2, (32, 32))
#                 felan1 = cv2.resize(sal, (32, 32))
#                 mahla = felan12.reshape(felan12.shape+(1,))
#                 # print("I o J",i,j)
#                 prediictlist.append(mahla)
#                 sam1 = felan12.reshape((1,) +felan12.shape+(1,))
#                 sam2 = felan1.reshape((1,) + felan1.shape+(1,))
#                 # print("saeedsaeed lklklkkl",sam1.shape,sam2.shape)
#                 # print("========>>>",counter,np.argmax(new_model1.predict(sam1)[0]),np.argmax(new_model1.predict(sam2)[0]))
#                 # #print(i, j)
#                 # plt.imshow(cv2.cvtColor(img[minbala + bala:minpaeen - paeen, i:j], cv2.COLOR_BGR2RGB))
#
#                 felan2 = cv2.resize(felan1, (400, 400))
#
#             else:
#
#                 min1 = np.min(img[minbala :minpaeen , i-2:j+2])
#                 max1 = np.max(img[minbala :minpaeen, i-2:j+2])
#
#                 min2 = np.min(img[:, i:j])
#                 max2 = np.max(img[:, i:j])
#                 # print("I o J",i,j)
#
#                 sal = ((img[minbala :minpaeen, i-2:j+2] - min1) / (max1 - min1)) * 255
#                 sal2 = ((img[:, i:j] - min2) / (max2 - min2)) * 255
#
#                 felan12 = cv2.resize(sal2, (32, 32))
#                 felan1 = cv2.resize(sal, (32, 32))
#                 mahla = felan12.reshape(felan12.shape+(1,))
#                 prediictlist.append(mahla)
#                 sam1 = felan12.reshape((1,) + felan12.shape+(1,))
#                 sam2 = felan1.reshape((1,) + felan1.shape+(1,))
#
#                 felan2 = cv2.resize(felan1, (400, 400))
#
#
#         elif counter == 2:
#             j+= 25
#             min1 = np.min(img[minbala:minpaeen, i - 2:j + 2])
#             max1 = np.max(img[minbala:minpaeen, i - 2:j + 2])
#             # print("I o J", i, j)
#             min2 = np.min(img[:, i:j])
#             max2 = np.max(img[:, i:j])
#
#             sal = ((img[minbala:minpaeen, i - 2:j + 2] - min1) / (max1 - min1)) * 255
#             sal2 = ((img[:, i:j] - min2) / (max2 - min2)) * 255
#
#             felan12 = cv2.resize(sal2, (32, 32))
#             felan1 = cv2.resize(sal, (32, 32))
#             sam1 = felan12.reshape((1,) + felan12.shape + (1,))
#             sam2 = felan1.reshape((1,) + felan1.shape + (1,))
#             # print("saeedsaeed lklklkkl",sam1.shape,sam2.shape)
#             # print("========>>>>>>", counter, np.argmax(new_model1.predict(sam1)[0]),np.argmax(new_model1.predict(sam2)[0]))
# # name = where + "/" + str(counter) + "_" + str(np.argmax(new_model1.predict(sam1)[0])) + ".bmp"
# # cv2.imwrite(name, felan1)
# # name = where + "/" + "Complete" + str(counter) + "_" + str(np.argmax(new_model1.predict(sam1)[0])) + ".bmp"
# # cv2.imwrite(name, felan12)
#
#
#         tdakheliakhari = time.time()
#         # print("Time",tdakheliakhari - tdakheli,len(prediictlist),counter)
#         counter += 1
#     tt4 = time.time()
#     amir = new_model1(np.reshape(prediictlist, (len(prediictlist),32, 32,1))).numpy().argmax(axis = 1)[:,None]
#     # print(amir)
#     tt3 = time.time()
#     # print("second phase", tt3 - tt2,tt3-tt4)
#     # print("FORRRRR partioning : ", t2)
#
#     # plt.imshow(img.astype('uint8'),cmap='gray', vmin=0, vmax=255)
#     # plt.show()
#     index.index("Number",amir)


def finncc(imgs):
    for img in imgs:
        img = json.loads(img)
        tt1 = time.time()
        min1 = np.min(img)
        max1 = np.max(img)
        img = ((img - min1) / (max1 - min1)) * 255
        #
        # plt.imshow(img.astype('uint8'),cmap='gray', vmin=0, vmax=255)
        # plt.show()

        flag1 = 0
        vahid = ""
        # print("-0-0-0-0 ", img.shape)
        # print(np.max(img))
        t01 = time.time()
        for tol in range(0,img.shape[1]):
            if flag1:
                break
            for s in range(0,img.shape[0]):
                if img[s, tol] > 65:
                    con11 = 0
                    con21 = 0
                    for s1 in range(0, int(img.shape[0] * 0.5)):
                        # if img[s1, tol-3] > 83:
                        if img[s1, tol-2] > 60:
                            con21 += 1
                        else:
                            con11 += 1
                    # print(con21 / (con21 + con11))
                    if con21 / (con21 + con11) > 0.3:
                        sahand = tol
                        flag1 = 1
                        break
        flag1 = 0
        for tol in range(img.shape[1]-1,0,-1):
            if flag1:
                break
            for s in range(0,int(img.shape[0]*0.3)):
                if img[s, tol] > 70:
                    lastsahand = tol
                    flag1 = 1
                    # print(lastsahand)
                    break
            aya = 0
            if flag1 ==1:
                con11 = 0
                con21 = 0
                for s1 in range(0,int(img.shape[0]*0.5)):
                    # print(img[s1, tol],s1)
                    # if img[s1, tol-3] > 83:
                    if img[s1 , tol ] > 70:
                        con21 += 1
                    else:
                        con11 += 1
                # print(con21 /(con21 + con11))
                if con21 /(con21 + con11) > 0.3:
                    lastsahand = tol
                    aya = 1
                        # break
            if aya == 0:
                flag1 = 0
        # print("===>",sahand,lastsahand)
        t2 = time.time() - t01

        img = cv2.resize(img[:, sahand:lastsahand], (250, 60))

        tt2 = time.time()
        # print("first phase", tt2 - tt1)
        # print(np.max(img))

        # plt.imshow(img.astype('uint8'),cmap='gray', vmin=0, vmax=255)
        # plt.show()

        counter = 0
        prediictlist = []

        for i in range(0 + 22, img.shape[1], 25):
            t01 = time.time()
            if counter >= 9:
                break
            tdakheli = time.time()
            j = i + 25
            minbala = 0
            minpaeen = 500
            for t in range(0, img.shape[0]):
                if img[t, i] > 50:
                    flag = 1
                    minbala = t
                    break
            for t in range(img.shape[0] - 1, 0, -1):
                if img[t, i] > 50:
                    flag = 1
                    minpaeen = t
                    break
            t2 += time.time() - t01
            # print("---------------------------")
            # tvasat = time.time() - tdakheli
            if counter in [0, 1, 4, 5, 6, 7, 8]:
                if counter > 6:
                    i = i + 3
                    j = j + 3
                    min1 = np.min(img[minbala:minpaeen, i:j])
                    max1 = np.max(img[minbala:minpaeen, i:j])

                    min2 = np.min(img[:, i:j])
                    max2 = np.max(img[:, i:j])

                    # img[minbala + 10:minpaeen-2, i:j] = ((img[minbala + 10:minpaeen-2, i:j] - min1) / (max1 - min1)) * 255
                    sal = ((img[minbala:minpaeen, i:j] - min1) / (max1 - min1)) * 255
                    sal2 = ((img[:, i:j] - min2) / (max2 - min2)) * 255

                    felan12 = cv2.resize(sal2, (32, 32))
                    felan1 = cv2.resize(sal, (32, 32))
                    mahla = felan12.reshape(felan12.shape + (1,))
                    # print("I o J", i, j)
                    prediictlist.append(mahla)
                    sam1 = felan12.reshape((1,) + felan12.shape + (1,))
                    sam2 = felan1.reshape((1,) + felan1.shape + (1,))
                    # print("saeedsaeed lklklkkl",sam1.shape,sam2.shape)
                    # print("========>>>",counter,np.argmax(new_model1.predict(sam1)[0]),np.argmax(new_model1.predict(sam2)[0]))
                    # #print(i, j)
                    # plt.imshow(cv2.cvtColor(img[minbala + bala:minpaeen - paeen, i:j], cv2.COLOR_BGR2RGB))

                    felan2 = cv2.resize(felan1, (400, 400))

                else:

                    min1 = np.min(img[minbala:minpaeen, i - 2:j + 2])
                    max1 = np.max(img[minbala:minpaeen, i - 2:j + 2])

                    min2 = np.min(img[:, i:j])
                    max2 = np.max(img[:, i:j])
                    # print("I o J", i, j)

                    sal = ((img[minbala:minpaeen, i - 2:j + 2] - min1) / (max1 - min1)) * 255
                    sal2 = ((img[:, i:j] - min2) / (max2 - min2)) * 255

                    felan12 = cv2.resize(sal2, (32, 32))
                    felan1 = cv2.resize(sal, (32, 32))
                    mahla = felan12.reshape(felan12.shape + (1,))
                    prediictlist.append(mahla)
                    sam1 = felan12.reshape((1,) + felan12.shape + (1,))
                    sam2 = felan1.reshape((1,) + felan1.shape + (1,))

                    felan2 = cv2.resize(felan1, (400, 400))


            elif counter == 2:
                j += 25
                min1 = np.min(img[minbala:minpaeen, i - 2:j + 2])
                max1 = np.max(img[minbala:minpaeen, i - 2:j + 2])
                # print("I o J", i, j)
                min2 = np.min(img[:, i:j])
                max2 = np.max(img[:, i:j])

                sal = ((img[minbala:minpaeen, i - 2:j + 2] - min1) / (max1 - min1)) * 255
                sal2 = ((img[:, i:j] - min2) / (max2 - min2)) * 255

                felan12 = cv2.resize(sal2, (32, 32))
                felan121 = felan12.reshape((1,) + felan12.shape + (1,))
                felan1 = cv2.resize(sal, (32, 32))
                sam1 = felan12.reshape((1,) + felan12.shape + (1,))
                sam2 = felan1.reshape((1,) + felan1.shape + (1,))
                # print("saeedsaeed lklklkkl",sam1.shape,sam2.shape)
                # print("========>>>>>>", counter, np.argmax(new_model1.predict(sam1)[0]),np.argmax(new_model1.predict(sam2)[0]))
            # name = where + "/" + str(counter) + "_" + str(np.argmax(new_model1.predict(sam1)[0])) + ".bmp"
            # cv2.imwrite(name, felan1)
            # name = where + "/" + "Complete" + str(counter) + "_" + str(np.argmax(new_model1.predict(sam1)[0])) + ".bmp"
            # cv2.imwrite(name, felan12)

            tdakheliakhari = time.time()
            # print("Time", tdakheliakhari - tdakheli, len(prediictlist), counter)
            counter += 1
        tt4 = time.time()
        # print("Shahid hemmmat", horoooof[model2(felan121).numpy().argmax(axis=1)[:, None][0][0]])
        # print(new_model1(np.reshape(prediictlist, (len(prediictlist),32, 32,1))).numpy().argmax(axis = 1)[:,None][0])
        retStr = ""
        contere = 0
        for i in new_model1(np.reshape(prediictlist, (len(prediictlist), 32, 32, 1))).numpy().argmax(axis=1)[:, None]:
            # print(i)
            if contere == 2:
                shsh = time.time()
                retStr = retStr + " " + horoooof[aplphabet(felan121).numpy().argmax(axis=1)[:, None][0][0]] + " "
                # print(time.time() - shsh)
            elif contere == 5:
                retStr = retStr + " "
            retStr = retStr + str(i[0])
            contere += 1
        # print(retStr)
        tt3 = time.time()
        # print("second phase", tt3 - tt2, tt3 - tt4)
        # print("FORRRRR partioning : ", t2)
        index.index("Number", retStr)
        # return retStr
        # plt.imshow(img.astype('uint8'),cmap='gray', vmin=0, vmax=255)
        # plt.show()
        # return vahid

