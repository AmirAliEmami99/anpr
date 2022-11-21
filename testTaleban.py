import Logic
import cv2

tempimage = cv2.imread("testImage.jpg")
result = Logic.anprApi(tempimage)

print('***********')
print(result)