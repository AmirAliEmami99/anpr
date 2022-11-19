import imp
# from re import S
import plateFinder
import getFrame
import charFinder1
import charFinder2
import charFinder3
import charFinder4
import charFinder5

# import charDetection
# import tracking
import cv2
import numpy as np
import time 

# ca = getFrame.Start_CSI(1640,1232)

#print(time.time())
start = time.time()
# READ THE IMAGE 


def anprApi(tempimage):
    # tempimage = cv2.imread("testImage.jpg")
    mainimg, modelimg = plateFinder.Image_prepareation(tempimage)
    afterRead = time.time()

    tedad = 500
    for i in range(0,1):
        # RESIZE THE IMAGE 
        # PASS TO PLATEFINDER MODEL
        # mainimg = getFrame.CSI_frame(1640,1232,ca)
        mainimg, modelimg = plateFinder.Image_prepareation(mainimg)
        # modelimg = mainimg.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        # #print("secondly",im.shape)
        # modelimg = np.ascontiguousarray(modelimg)
        #print("shape of captured",modelimg.shape,mainimg.shape,type(modelimg),type(mainimg))
        platesReadyForOCR128Y640X , NumOfPlates, platesCoordinates = plateFinder.run(mainimg,modelimg)
        #print("number of plates :" , NumOfPlates)
        # PREPARE THE PLATES FOR OCR
        if NumOfPlates == 0:
            continue
        vasatesh = time.time()
        platesReadyForOCR128Y640X = platesReadyForOCR128Y640X.transpose((2, 0, 1))[::-1]
        platesReadyForOCR128Y640X = np.ascontiguousarray(platesReadyForOCR128Y640X)
        # PASS THE PLATES TO OUR OCR MODEL
        #print("shape of outputshape:",platesReadyForOCR128Y640X.shape,NumOfPlates)
        print(NumOfPlates)

        if NumOfPlates==1:
            OCRed = charFinder1.run(platesReadyForOCR128Y640X,platesReadyForOCR128Y640X)
        elif NumOfPlates==2:
            OCRed = charFinder2.run(platesReadyForOCR128Y640X,platesReadyForOCR128Y640X)
        elif NumOfPlates==3:
            OCRed = charFinder3.run(platesReadyForOCR128Y640X,platesReadyForOCR128Y640X)
        elif NumOfPlates==4:
            OCRed = charFinder4.run(platesReadyForOCR128Y640X,platesReadyForOCR128Y640X)
        elif NumOfPlates==5:
            OCRed = charFinder5.run(platesReadyForOCR128Y640X,platesReadyForOCR128Y640X)
        print(i)
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print(OCRed)
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print(platesCoordinates)

        ocLast = []
        ghaleb = []
        topLeft = []
        buttomRight = []
        plateConfidence = []

        

        for i in range(len(OCRed)):
            print(i)
            

            if OCRed[i] == 'notRead':
                print('notRead')
                pass

            else:
                
                
                ghaleb.append(OCRed[i])

                print(platesCoordinates[i])
                print(platesCoordinates[i][:2])
                topLeft = platesCoordinates[i][:2]
                ghaleb.append(topLeft)

                print(platesCoordinates[i][2:4])
                buttomRight = platesCoordinates[i][2:4]
                ghaleb.append(buttomRight)

                print(platesCoordinates[i][4])
                plateConfidence = platesCoordinates[i][4]
                ghaleb.append(plateConfidence)

                print('#########')
                print(ghaleb)
                ocLast.append(ghaleb)
                ghaleb = []
                print('#########')
            
        print('@@@@@@@')
        print(ocLast)
        print('@@@@@@@')

        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    return ocLast
    

#print(OCRed)


# end = time.time()
# print(end - start, (end - afterRead)/tedad,afterRead - start,vasatesh - afterRead, end - vasatesh)
#print(end - start, (end - afterRead)/tedad,afterRead - start)
























