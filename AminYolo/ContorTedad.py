# import json
# import os
# import subprocess
# import ConFindercopy
import cv2
# import Partioningcopy
import numpy as np
import tensorflow as tf
import time
# import ReadImage
# import status
# import GUI as G
# import pickle
import imutils
import index
import copy
from matplotlib import pyplot as plt


new_model1 = tf.keras.models.load_model("zeroToNineOurDataset.h5")
aplphabet = tf.keras.models.load_model('horof.h5')
new_model = tf.keras.models.load_model('pNCompleteDatasetModel.h5')
horoooof = ["Alef", "Be", "Pe", "Te", "Se3Noghte", "jim", "Dal", "Zhe","Sin", "She", "Sad", "TeDasteDar", "Ein", "Phe", "ghaf", "Lam", "mim","Non", "Vav", "He", "ie"]

path34 = "Tessting"
lastdir = {}
Platetimes = 0
def ReadImage1(path):
    return cv2.imread(path)

def index(jens,data,trustedcon = 0):
    global Platetimes
    if jens == "IMG":
        pass
        protot(data)
    elif jens == "CON":
        # print(len(data), "kolesh injast")
        PorNP(data,trustedcon)
    elif jens == "TrustedCON":
        Platetimes = len(trustedcon)
        print(len(trustedcon), "kolesh injast")

        # draws(trustedcon)
        # finncc(data)
    elif jens == "Number":
        print(time.time())
        lastdir[path34]["number"].append(data)
        print("pelak",data)
def draws(col):
    print("length",len(lastdir[path34]["mainimg"]))
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    ila = lastdir[path34]["mainimg"]
    cv2.drawContours(ila, col, -1, (0, 255, 0), 3)
    lastdir[path34]["drawedimg"] = ila
    # plt.imshow(cv2.cvtColor(ila, cv2.COLOR_BGR2RGB))
    # plt.show()
def finncc(imgs):
    # img = json.loads(img[0])
    lastdic = {}
    for img in imgs:
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
                if img[s, tol] > 60:
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
                    if img[s1 , tol ] > 60:
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
        index("Number", retStr,)
        # return retStr
        # plt.imshow(img.astype('uint8'),cmap='gray', vmin=0, vmax=255)
        # plt.show()
        # return vahid
def protot(img):
    print("Confinder Started")
    t01 = time.time()
    saeed = []
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.show()
    # print("imageeedirectory",img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print("CONTUR0.5 : ", time.time() - t01)
    img12 = copy.deepcopy(gray)
    # print("CONTUR1 : ",time.time() - t01)
    for t in range(0,2):
        if t==0:
            # continue
            edged = cv2.Canny(gray, 30, 100)  # Edge detection
            # print("CONTUR1.3 : ", time.time() - t01)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            # print("CONTUR1.5 : ", time.time() - t01)
            edged = cv2.dilate(edged, kernel, iterations=0)
            # print("CONTUR2 : ", time.time() - t01)

        else:
        #     #Option-shab
        #     #print("dark")
        #     gray = cv2.equalizeHist(gray)
        #
            # Option-Roz
            # min1 = np.min(gray)
            # max1 = np.max(gray)
            # min1 = np.min(gray) +30
            # max1 = np.max(gray)
            gray = img12
            gray = np.clip(gray, np.min(gray) +90, np.max(gray))
            # gray = np.clip(gray, 140, 160)
            min1 = np.min(gray)
            max1 = np.max(gray)
            gray = ((gray - min1) / (max1 - min1))  * 250 + 4
            # gray = 255 - gray
            print("CONTUR3-0.5 : ", time.time() - t01)
            # edged = cv2.Canny(gray.astype("uint8"), 30, 35)  # Edge detection
            edged = cv2.Canny(gray.astype("uint8"), 100, 220)  # Edge detection

            print("CONTUR3-0.75 : ", time.time() - t01)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            print("CONTUR3-0.85 : ", time.time() - t01)
            edged = cv2.dilate(edged, kernel, iterations=1)
            print("CONTUR3 : ", time.time() - t01)

        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print("CONTUR3.2 : ", time.time() - t01)
        contours = imutils.grab_contours(keypoints)
        # print("CONTUR3.4 : ", time.time() - t01)
        # contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        print("inja",type(contours),len(contours))

        # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
        # plt.imshow(gray.astype('uint8'), cmap='gray', vmin=0, vmax=255)
        # plt.show()
        # plt.imshow(img.astype('uint8'), cmap='gray', vmin=0, vmax=255)
        # plt.show()

        # print("CONTUR4 : ", time.time() - t01)
        # print(type(contours),len(contours))
        # contours = np.array(contours)
        # print(contours)
        # print(contours.shape,type(contours),len(contours))
        # saeed = np.apply_along_axis(applyim, 0, contours)
        # print(saeed.tolist())

        # sha =  time.time()
        # p = Process(target=applyim, args=(contours[0],img12))
        # p.start()
        # print("hafez asad",time.time() - sha)
        # p.join()



        # a_pool = multiprocessing.Pool()
        # saeed = a_pool.map(applyim, contours)
        # a_pool.close()

        # print("shaid hemmat",khoroji)
        # print("TEDAD HAIE CONTOR",len(contours))
        outputcontur = contours
        for contour in contours:


            x,y,w,h = cv2.boundingRect(contour)
            area = w*h


            print(area,w/h)
            if 31000> area >2000 and 1<= w/h <5.2:
                if w/h < 4.77:
                    cropped_image = img12[y:y + h, x-7:x + int(5*h)]
                    saeed.append([x,y,w,h,cropped_image,contour])
                    print("shod")
                else:
                    cropped_image = img12[y:y + h, x:x + w]
                    saeed.append([x,y,w,h,cropped_image,contour])
                    print("shod")

                # PorNP([cropped_image], [contour])
                # approx = cv2.approxPolyDP(contour, 10, True)
                # hamed = cv2.boundingRect(approx)
                # # print("mmmmmmmmmax", w / h, "area", area, )
                # cropped_image = img12[hamed[1]:hamed[1] + hamed[3], hamed[0]:hamed[0] + hamed[2]]
                # plt.imshow(cropped_image.astype('uint8'), cmap='gray', vmin=0, vmax=255)
                # plt.show()

        # print("CONTUR5 : ", time.time() - t01)

    ko = []
    han = []

    for i in range(0,len(saeed)):
        if saeed[i] == 0:
            continue
        # print("jan fadaee",saeed[i][0])
        if i in ko:
            continue
        if i in han:
            continue
        max1 = 0
        adad = i
        for j in range(i+1,len(saeed)):
            if saeed[j] == 0:
                continue
            if j in ko:
                continue
            if j in han:
                continue
            # print(saeed[i][1],saeed[j][1])
            # print((saeed[i][0]-saeed[j][0])**2 + (saeed[i][1]-saeed[j][1])**2)
            # plt.imshow(cv2.cvtColor(saeed[i][4], cv2.COLOR_BGR2RGB))
            # plt.show()
            # plt.imshow(cv2.cvtColor(saeed[j][/4], cv2.COLOR_BGR2RGB))
            # plt.show()
            if (saeed[i][0]-saeed[j][0])**2 + (saeed[i][1]-saeed[j][1])**2 < 10000:
                if saeed[j][2] * saeed[j][3]  > saeed[i][2] * saeed[i][3] and saeed[j][2] * saeed[j][3] > max1 :
                    max1 = saeed[j][2] * saeed[j][3]
                    adad = j
                else:
                    ko.append(j)
            # print(ko,han)
        #     print("-------------")
        # print("-><><><><><><><>--")
        han.append(adad)
    # print("CONTUR5.3 : ", time.time() - t01)
    # print(han)
    outputlist = []
    outputcontur = []
    print("hamaashon", len(han),len(ko))
    for i in han:
        outputlist.append(saeed[i][4])
        outputcontur.append(saeed[i][5])


    # for i in range(0,len(saeed)):
    #     if len(saeed[i][4][1]) == 0:
    #         continue
    #     print("kapio",len(saeed[i][4][1]))
    #     print(saeed[i][2]*saeed[i][3])
    #     # PorNP([saeed[i][4]],[saeed[i][5]])
    #     # plt.imshow(saeed[i][4].astype('uint8'), cmap='gray', vmin=0, vmax=255)
    #     # plt.show()
    #
    #     outputlist.append(saeed[i][4])
    #     outputcontur.append(saeed[i][5])
    # print("hagh",len(outputlist))
    neshone = 0
    # for i in outputlist:
    #     print("chandtast ??????????")
    #     # try:
    #     print()
    #     print("javab paeenias",PorNP([i],[outputcontur[neshone]]))
    #     plt.imshow(i.astype('uint8'),cmap='gray', vmin=0, vmax=255)
    #     plt.show()
    #
    #     # input()
    #     # except:
    #     #     print("inja gir karde")
    #     neshone+=1

    print("Confinder Finish")
    # print(outputlist)
    # print(outputcontur)
    index("CON",outputlist,outputcontur)
    # return outputlist
def PorNP(Contur,col):
    # print(Contur)
    global lastdir
    print("PLate or not plate Started",len(Contur))
    res = []
    shahid = []
    for i in Contur:
        # print("saeeedsaeeedsaeeedsaeeedsaeeedsaeeedsaeeedsaeeedsaeeed",i.shape[1]/i.shape[0])
        try:

            input_array = cv2.resize(i,(32,32))
            input_array = np.reshape(input_array, (32, 32,1))
            shahid.append(input_array)

        except :

            print("rad Shude!")
    # input_array = np.expand_dims(input_array, axis=0)
        # print("havij",input_array.shape)
        # print(new_model.predict(input_array)[0][0])
    # print(len(shahid))
    shash = []
    trustfordraw = []
    sad = 0
    # print(new_model.predict(np.reshape(shahid, (len(shahid),32, 32,1))))
    print("peida shude ha",len(col))
    shal = new_model(np.reshape(shahid, (len(shahid),32, 32,1))).numpy()
    print(shal)
    for am in shal:
        # am = am.numpy()

        # print("ista",am < 0.5,am[0],[am,am])
        if am[0] > 0.3 and i.shape[1] / i.shape[0] < 3.9:
            res.append(1)
            print("pelak nist")
        else:
            print("amamamamam",am)
            if am[0] < 0.3:
                print("pelak hast")
                shash.append(Contur[sad])
                trustfordraw.append(col[sad])

                # lists = Contur[sad].tolist()
                # json_str = json.dumps(lists)
                # shash.append(json_str)
            res.append(am[0])
        sad += 1
        # age w/h bod bozorg taresh o bar midarim
    print("shahhahahah",len(trustfordraw))

    print("PLate or not plate Started")
    lastdir[path34]["pelakha"] = shash
    index("TrustedCON",shash,trustfordraw)

    # return shash
def ReadReq(listOfDir):
    shaf = 0
    for i in range(0,1):
        inja = time.time()
        for image_path in listOfDir:
            sqpc= time.time()
            print(inja)
            if image_path == ".DS_Store":
                continue
            input_path1 = image_path
            print(input_path1)
            img = ReadImage1(input_path1)
            global path34
            path34 = image_path
            lastdir[path34] = {"mainimg": img}
            lastdir[path34]["number"] = []
            print(time.time() - inja)
            sa1 = time.time()
            shaf += 1

            index("IMG",img)
# ReadReq(["Tessting/2021-01-16_05-17-18-241_26D66820-IRN.bmp","Tessting/2021-01-15_14-15-33-678_55C99110-IRN.jpg"])
# ReadReq(["/Users/amirsajjad/Desktop/CarID_OCR/fashtkhudam/12C31911-IRN_2021-09-04_10-46-17-529.jpg"])
# ReadReq(["/Users/amirsajjad/Desktop/CarID_OCR/fashtkhudam/34214521-IRN_2021-09-02_14-22-54-773.jpg"])
# /Users/amirsajjad/Desktop/CarID_OCR/fashtkhudam/34214521-IRN_2021-09-02_14-22-54-773.jpg
# /Users/amirsajjad/Desktop/CarID_OCR/fashtkhudam/12C31911-IRN_2021-09-04_10-46-17-529.jpg
# 27H18766-IRN_2021-09-02_14-15-18-098.jpg
# import os
# casp = 0
# for i in os.listdir("/Users/amirsajjad/Desktop/CarID_OCR/fashtkhudam"):
#     print(i)
#
#     # casp += 1
#     # if casp <26 :
#     #     continue
#
#     pa = "/Users/amirsajjad/Desktop/CarID_OCR/fashtkhudam/"+str(i)
#     ReadReq([pa])