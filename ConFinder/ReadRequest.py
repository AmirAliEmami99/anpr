import json
import os

import cv2

import time

import imutils
# import index
import copy

# new_model1 = tf.keras.models.load_model("zeroToNineOurDataset.h5")
# aplphabet = tf.keras.models.load_model('horof.h5')
# new_model = tf.keras.models.load_model('pNCompleteDatasetModel.h5')
# horoooof = ["Alef", "Be", "Pe", "Te", "Se3Noghte", "jim", "Dal", "Zhe","Sin", "She", "Sad", "TeDasteDar", "Ein", "Phe", "ghaf", "Lam", "mim","Non", "Vav", "He", "ie"]

pathRoReadfromdir = "Tessting"
ash = 0
sa = 0
def ReadImage1(path):
    return cv2.imread(path)

def index(jens,data):
    if jens == "IMG":
        pass
        protot(data)
    elif jens == "CON":
        pass
        # print(data)
def protot(img):
    print("Confinder Started")
    t01 = time.time()
    saeed = []
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.show()
    # print("imageeedirectory",img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(gray,type(gray))
    # print("CONTUR0.5 : ", time.time() - t01)
    img12 = copy.deepcopy(gray)
    # print("CONTUR1 : ",time.time() - t01)
    for t in range(0,1):
        if t==0:
            edged = cv2.Canny(gray, 30, 100)  # Edge detection
            # print("CONTUR1.3 : ", time.time() - t01)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            # print("CONTUR1.5 : ", time.time() - t01)
            edged = cv2.dilate(edged, kernel, iterations=1)
            # print("CONTUR2 : ", time.time() - t01)

        # else:
        #     #Option-shab
        #     #print("dark")
        #     # gray = cv2.equalizeHist(gray)
        #
        #     # Option-Roz
        #     min1 = np.min(gray)
        #     max1 = np.max(gray)
        #     gray = ((gray - min1) / (max1 - min1)) * 255
        #
        #     print("CONTUR3-0.5 : ", time.time() - t01)
        #     edged = cv2.Canny(gray.astype("uint8"), 170, 200)  # Edge detection
        #     print("CONTUR3-0.75 : ", time.time() - t01)
        #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        #     print("CONTUR3-0.85 : ", time.time() - t01)
        #     edged = cv2.dilate(edged, kernel, iterations=2)
        #     print("CONTUR3 : ", time.time() - t01)

        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print("CONTUR3.2 : ", time.time() - t01)
        contours = imutils.grab_contours(keypoints)
        # print("CONTUR3.4 : ", time.time() - t01)
        # contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[0:20]
        # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
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
        for contour in contours:
            # approx = cv2.approxPolyDP(contour, 10,True)
            x,y,w,h = cv2.boundingRect(contour)
            area = w*h

            if 27000> area >1000 and 2.8<= w/h <5.2:
                if w/h < 4.77:
                    cropped_image = img12[y:y + h, x-7:x + int(5*h)]
                    saeed.append([x,y,w,h,cropped_image])
                else:
                    cropped_image = img12[y:y + h, x:x + w]
                    saeed.append([x,y,w,h,cropped_image])

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
    for i in han:
        outputlist.append(saeed[i][4])
    print("Confinder Finish")
    index("CON",outputlist)
    # return outputlist


shaf = 0
for i in range(0,1):
    inja = time.time()
    for image_path in os.listdir(pathRoReadfromdir):
        sqpc= time.time()
        print(inja)
        if image_path == ".DS_Store":
            continue
        input_path1 = os.path.join(pathRoReadfromdir, image_path)
        print(input_path1)
        img = ReadImage1(input_path1)
        print(time.time() - inja)
        sa1 = time.time()
        shaf += 1
        index("IMG",img)
        sa = sa + time.time() - sa1
    ash += time.time() - inja
print(shaf)
print(ash)



