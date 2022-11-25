import Logic
import cv2

imgpath = 'imagesForTest/2021-01-14_13-59-36-617_42L56477-IRN.jpg'
img = cv2.imread(imgpath)


javab = Logic.anprApi(img)

print(javab)