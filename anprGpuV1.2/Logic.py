# import imp
# from re import S
import plateFinderBatch
# import getFrame
import charFinderBatch
# import charFinder1
# import charFinder2
# import charFinder3
# import charFinder4
# import charFinder5

# import charDetection
# import tracking
import cv2
import numpy as np
import time

timeStart = time.time()


def anprApi(inputImage):
    # resize the input image for plate finder model
    mainimg, modelimg = plateFinderBatch.Image_prepareation(inputImage)

    timeReadAndPreparation = time.time()

    # pass the main image and prepared image to the plate finder model
    platesReadyForOCR128Y640X, NumOfPlates, platesCoordinates = plateFinderBatch.run(mainimg, modelimg)

    timeplateFinderBatch = time.time()

    if NumOfPlates == 0:
        print('No Plate Detected !!!')
        return []

    else:
        OCRed = charFinderBatch.run(platesReadyForOCR128Y640X, platesReadyForOCR128Y640X)
        print(OCRed)
        # return OCRed
        timecharFinderBatch = time.time()

        # PASS THE PLATES TO OUR OCR MODEL
        # print("shape of outputshape:",platesReadyForOCR128Y640X.shape,NumOfPlates)
        # if NumOfPlates==1:
        #     OCRed = charFinder1.run(platesReadyForOCR128Y640X,platesReadyForOCR128Y640X)
        # elif NumOfPlates==2:
        #     OCRed = charFinder2.run(platesReadyForOCR128Y640X,platesReadyForOCR128Y640X)
        # elif NumOfPlates==3:
        #     OCRed = charFinder3.run(platesReadyForOCR128Y640X,platesReadyForOCR128Y640X)
        # elif NumOfPlates==4:
        #     OCRed = charFinder4.run(platesReadyForOCR128Y640X,platesReadyForOCR128Y640X)
        # elif NumOfPlates==5:
        #     OCRed = charFinder5.run(platesReadyForOCR128Y640X,platesReadyForOCR128Y640X)

        # PRINT OUT THE RESULT

        global ocLast
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

                # print(platesCoordinates[i])
                # print(platesCoordinates[i][:2])
                topLeft = platesCoordinates[i][:2]
                ghaleb.append(topLeft)

                # print(platesCoordinates[i][2:4])
                buttomRight = platesCoordinates[i][2:4]
                ghaleb.append(buttomRight)

                # print(platesCoordinates[i][4])
                plateConfidence = platesCoordinates[i][4]
                ghaleb.append(plateConfidence)

                # print('#########')
                # print(ghaleb)
                ocLast.append(ghaleb)
                ghaleb = []
                # print('#########')

        # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        # print(ocLast)
        # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        return ocLast


timeEnd = time.time()
