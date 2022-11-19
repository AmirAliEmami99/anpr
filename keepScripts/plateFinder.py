import numbers
import os
import platform
import sys
from pathlib import Path
# from time import time
import time
import torch
import numpy as np
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utilsk.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utilsk.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utilsk.plots import Annotator, colors, save_one_box
from utilsk.torch_utils import select_device, smart_inference_mode
from utilsk.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
                                 cutout, letterbox, mixup, random_perspective)

def Set_params(we,so):
    global weights, source, data,imgsz,conf_thres,iou_thres,max_det,device,view_img,save_txt,save_conf, save_crop, nosave, classes,  agnostic_nms,augment,visualize,update,project, name,exist_ok, line_thickness,  hide_labels,  hide_conf, half, dnn, vid_stride
    weights = we
    source = so
    data= "plateDatasets.yaml"  # dataset.yaml path
    imgsz=(416, 416)  # inference size (height, width)
    conf_thres=0.25  # confidence threshold
    iou_thres=0.45  # NMS IOU threshold
    max_det=1000 # maximum detections per image
    device=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False  # show results
    save_txt=False # save results to *.txt
    save_conf=False  # save confidences in --save-txt labels
    save_crop=False  # save cropped prediction boxes
    nosave=False  # do not save images/videos
    classes=None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False  # class-agnostic NMS
    augment=False  # augmented inference
    visualize=False  # visualize features
    update=False  # update all models
    project= 'runs/detect' # save results to project/name
    name='exp'  # save results to project/name
    exist_ok=False  # existing project/name ok, do not increment
    line_thickness=3  # bounding box thickness (pixels)
    hide_labels=False  # hide labels
    hide_conf=False  # hide confidences
    half=False  # use FP16 half-precision inference
    dnn=False # use OpenCV DNN for ONNX inference
    vid_stride=1 # video frame-rate stride
def Image_prepareation(main_img):
    im = letterbox(main_img, 416, stride=416, auto=True)[0]
    # #print("secondly",im.shape)
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    # #print("secondly",im.shape)
    model_img = np.ascontiguousarray(im)
    return main_img, model_img
def Loadmodel():
    global device,model, stride,seen, windows, dt, names,pt,weights, source, data,imgsz,conf_thres,iou_thres,max_det,device,view_img,save_txt,save_conf, save_crop, nosave, classes,  agnostic_nms,augment,visualize,update,project, name,exist_ok, line_thickness,  hide_labels,  hide_conf, half, dnn, vid_stride

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    # imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
def run(
	    mainim,
        trueim,
):
    global weights, source, data,imgsz,conf_thres,iou_thres,max_det,device,view_img,save_txt,save_conf, save_crop, nosave, classes,  agnostic_nms,augment,visualize,update,project, name,exist_ok, line_thickness,  hide_labels,  hide_conf, half, dnn, vid_stride

    source = str(source)
    # save_txt=True
    # save_crop=True
    # save_img = not nosave and not source.endswith('.txt')  # save inference images
    # is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    # is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    # webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    # screenshot = source.lower().startswith('screen')


    # Directories
    # save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model

    im = trueim
    dataset = [0,im,mainim,None,""]
    im0s = mainim
    with dt[0]:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        # #print("ghable errore",im.shape)
        # Inference
        shahid = time.time()

        with dt[1]:
            # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=False)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        #print("timesh k shayad ziad bashe",time.time() - shahid)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        shahid1 = time.time()
        # Process predictions
        numsas = -1
        koli = 0
        # #print(pred)
        for i, det in enumerate(pred):  # per image
            # import time
            # shahid = time.time()
            # #print("pred",det,i)
            im0= im0s.copy()
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            # annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            # #print("timesh k shayad ziad bashe",time.time() - shahid)

            if len(det):
                # #print(det[0][0:4].round())
                xyxy = det[::][0:4].round()
                # Rescale boxes from img_size to im0 size

                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                # #print("timesh k shayad ziad bashe",time.time() - shahid)
                # shahid1 = time.time()
                print('###########################')   
                print(det)
                print('###########################')
                print(det.tolist())
                platesCoordinates = det.tolist()

                for *xyxy, conf, cls in reversed(det):

                    numsas += 1
                    if numsas == 0:
                        zahra = save_one_box(xyxy, imc, file="", BGR=False,save = False)
                        # #print(zahra.shape)
                        zahra = cv2.resize(zahra,(640,128))
                        koli = zahra
                    else:
                        shahid = time.time()
                        zahra = save_one_box(xyxy, imc, file="", BGR=False,save = False)
                        # #print(zahra.shape)
                        

                        zahra = cv2.resize(zahra,(640,128))
                        koli = np.concatenate((koli, zahra), axis=1)
                        # #print("\aaa timesh k shayad ziad bashe",time.time() - shahid)


                    # cv2.imwrite(str(numsas)+".png", zahra)
                    # cv2.imshow('image', zahra)
                    # cv2.waitKey(1000)
                    #cv2.imshow('image', koli)
                    #cv2.waitKey(1000)
        #print("timesh part2 k shayad ziad bashe",time.time() - shahid1)
        cv2.imshow('image', mainim)
        cv2.waitKey(100)
                
        return koli , numsas+1, platesCoordinates
        cv2.imwrite("aya.png", koli)



Set_params("best1.engine","NO")
Loadmodel()
# tempimage = cv2.imread("15D57811-IRN_2021-09-02_14-10-54-185.jpg")
# mainimg, modelimg = Image_prepareation(tempimage)
# platesReadyForOCR128Y640X = run(mainimg,modelimg)

# cv2.imshow('image', tempimage)
# cv2.waitKey(1000)