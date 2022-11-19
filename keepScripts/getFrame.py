import jetson_utils
import vpi
import numpy as np
import PIL.Image
import cv2

def Start_CSI(Width, Height):
    camera = jetson_utils.gstCamera(Width, Height, '0')
    camera.Open()
    return camera
def CSI_frame(Width, Height,camera):

    frame, width, height = camera.CaptureRGBA(zeroCopy=1)
    # cvtColor(frame, zahra, CV_RGBA2BGR)
    # zahra = frame.convert('RGB')
    FrameFromCsi = np.uint8(jetson_utils.cudaToNumpy(frame))[::,::,:3:]

    # FrameFromCsi = vpi.asimage(zahra)
    # with vpi.Backend.CUDA: 
    #     vpi.clear_cache()

    # opt 1 for return
    return FrameFromCsi

    # opt 2 for return
    # return FrameFromCsi, width, height

def Close_CSI(camera):
    camera.Close()

# ca = Start_CSI(416,416)
# for i in range(0,100):
#     javab = CSI_frame(416,416,ca)
#     print(type(javab),javab.shape)
#     cv2.imshow('image', javab)
#     cv2.waitKey(5)
