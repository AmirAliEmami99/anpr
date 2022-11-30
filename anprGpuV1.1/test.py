import Logic
import cv2

imgpath = 'imagesForTest/2021-01-14_13-59-36-617_42L56477-IRN.jpg'
imgpath = 'D:/resource/bugImages/2021-01-14_07-48-41-039_13823182-IRN.jpg'
imgpath = 'D:/resource/sepandDocument/imagesForTest/tracking/0.jpg'
img = cv2.imread(imgpath)
# cv2.imshow('test', img)


javab = Logic.anprApi(img)

print(javab)