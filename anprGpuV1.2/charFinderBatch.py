import numbers
import os
import platform
import sys
from pathlib import Path
import torch
import time
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
                                 cutout, letterbox, mixup, random_perspective)

namess = {10: "Alef",
          11: "Be",
          12: "Pe",
          13: "Te",
          14: "Se3Noghte",
          15: "jim",
          16: "Dal",
          17: "Zhe",
          18: "Sin",
          19: "She",
          20: "Sad",
          21: "TeDasteDar",
          22: "Ein",
          23: "Phe",
          24: "ghaf",
          25: "Lam",
          26: "mim",
          27: "Non",
          28: "Vav",
          29: "He",
          30: "ie"}


def Set_params(Modelpath, SourcePath, sizeY=128, fixsizeX=640):
    global weights, source, data, imgsz, conf_thres, iou_thres, max_det, device, view_img, save_txt, save_conf, save_crop, nosave, classes, agnostic_nms, augment, visualize, update, project, name, exist_ok, line_thickness, hide_labels, hide_conf, half, dnn, vid_stride
    weights = Modelpath
    source = SourcePath
    data = "plateDatasets.yaml"  # dataset.yaml path
    imgsz = (sizeY, fixsizeX)  # inference size (height, width)
    conf_thres = 0.3  # confidence threshold
    iou_thres = 0.45  # NMS IOU threshold
    max_det = 1000  # maximum detections per image
    device = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img = False  # show results
    save_txt = False  # save results to *.txt
    save_conf = False  # save confidences in --save-txt labels
    save_crop = False  # save cropped prediction boxes
    nosave = False  # do not save images/videos
    classes = None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms = False  # class-agnostic NMS
    augment = False  # augmented inference
    visualize = False  # visualize features
    update = False  # update all models
    project = 'runs/detect'  # save results to project/name
    name = 'exp'  # save results to project/name
    exist_ok = False  # existing project/name ok, do not increment
    line_thickness = 3  # bounding box thickness (pixels)
    hide_labels = False  # hide labels
    hide_conf = False  # hide confidences
    half = False  # use FP16 half-precision inference
    dnn = False  # use OpenCV DNN for ONNX inference
    vid_stride = 1  # video frame-rate stride


def Image_prepareation(main_img, imgsze):
    global imgsz
    imgsz = imgsze
    im = letterbox(main_img, imgsz, stride=32, auto=True)[0]
    # #print("secondly",im.shape)
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    # #print("secondly",im.shape)
    model_img = np.ascontiguousarray(im)
    return main_img, model_img


def Loadmodel():
    global device, model, stride, seen, windows, dt, names, pt, weights, source, data, imgsz, conf_thres, iou_thres, max_det, device, view_img, save_txt, save_conf, save_crop, nosave, classes, agnostic_nms, augment, visualize, update, project, name, exist_ok, line_thickness, hide_labels, hide_conf, half, dnn, vid_stride

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    # imgsz = check_img_size(imgsz, s=stride)  # check image size
    # print(pt)
    # Dataloader
    bs = 2  # batch_size
    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())


def run(
        mainim,
        trueim,
):
    global weights, source, data, imgsz, conf_thres, iou_thres, max_det, device, view_img, save_txt, save_conf, save_crop, nosave, classes, agnostic_nms, augment, visualize, update, project, name, exist_ok, line_thickness, hide_labels, hide_conf, half, dnn, vid_stride

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
    dataset = [0, im, mainim, None, ""]
    im0s = mainim
    # print("size of ture image", trueim.shape)
    with dt[0]:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        # #print("ghable errore",im.shape)
        # Inference
        shahid = time.time()
        # print("ghable run gereftan", im.shape)
        with dt[1]:
            # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
        # print(pred.size())
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        # print("CHARtimesh k shayad ziad bashe",time.time() - shahid)
        # print("len non_max_suppression:", len(pred))
        # print(pred[0])
        # print("-------------------")
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        numsas = -1
        koli = 0
        chagho = []
        for i, det in enumerate(pred):  # per image
            # im0, frame = im0s.copy(), getattr(dataset, 'frame', 0)
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # imc = im0.copy() if save_crop else im0  # for save_crop
            # annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            # det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            # #print results
            # #print("javab :",det[:, 5])
            # inja = det.numpy()
            # print(len(det),det.shape)
            # #print(det)

            shahid12 = time.time()
            det = det[det[:, 0].sort()[1]]
            # print(f"pred {i} is : {det}")

            # #print(det)
            # print("part2-sorting char k shayad ziad bashe",time.time() - shahid12)
            shahid = time.time()

            minak = 5000
            maxak = 0
            tartib = []

            javabeinja = 0
            strmikham = ""
            whereISalphabet = 0
            # try:
            minusplus = 0
            temppelak = ""
            temppelakarr = []

            # print(det)
            flag = 0
            doros = []
            adad = 0
            # print(det)
            # print(f'size of tensor: {det.size()}')
            # print(f'size of tensor: {det.size(dim=0)}')
            if det.size(dim=0) < 8:
                chagho.append('notRead')

            for t in det[:, 5]:
                # print(f'int(t) in det[:, 5]: {int(t)}')
                temppelakarr.append(int(t))

                # print(f'doros array: {doros}')

                if flag == 1:
                    minusplus += 1
                    doros.append(int(t))
                    if minusplus == 5:
                        if len(doros) == 8:
                            # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
                            chagho.append(doros)
                        # else:
                        #     print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                        #     chagho.append(['notRead'])
                        doros = []
                        flag = 0
                        minusplus = 0

                if int(t) >= 10 and flag == 0:
                    flag = 1
                    doros = temppelakarr[-3::]

            # #print(strmikham)
            # #print(chagho)
            # print("part2 char k shayad ziad bashe",time.time() - shahid)
        print('@@@@@@@@@@@@@@@@')
        print(chagho)
        return chagho
        # except:
        #     #print("error khurde !!")
        #     return "nashude"
        #     # #print results


# python3 detect.py --weights Charmodel.engine --source
# python3 detect.py --weights Charmodel.engine --img 640 --conf 0.6 --source "1.png"

#     # Write results
#     for *xyxy, conf, cls in reversed(det):
#         if save_txt:  # Write to file
#             xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
#             line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
#             with open(f'{txt_path}.txt', 'a') as f:
#                 f.write(('%g ' * len(line)).rstrip() % line + '\n')

#         # if save_img or save_crop or view_img:  # Add bbox to image
#         #     c = int(cls)  # integer class
#         #     label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
#         #     annotator.box_label(xyxy, label, color=colors(c, True))
#         save_crop = True
#         if save_crop:
#             c = int(cls)  # integer class
#             #print("injaaa",xyxy,type(imc))
#             zahra = save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=False)
#             imkochik = letterbox(zahra, new_shape=(64, 224), stride=32, auto=True)[0]
#             imkochik = imkochik.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
#             imkochik = np.ascontiguousarray(imkochik)
#             aadad = forchar.run(zahra,imkochik)
#             annotator.box_label(xyxy, aadad, color=colors(c, True))
#             #print("tekhe",aadad)
#             # c = int(cls)  # integer class
#             # label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
#             # annotator.box_label(xyxy, label, color=colors(c, True))

# # Stream results
# im0 = annotator.result()
# view_img = True
# if view_img:
#     if platform.system() == 'Linux' and p not in windows:
#         windows.append(p)
#         cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
#         cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
#     cv2.imshow(str(p), im0)
#     cv2.waitKey(1)  # 1 millisecond

Set_params("Charmodel.pt", "NO")
Loadmodel()
# tempimage = cv2.imread("aya1.png")
# tempimage = cv2.imread("aya.png")
# tempimage = tempimage.transpose((2, 0, 1))[::-1]
# tempimage = np.ascontiguousarray(tempimage)

# run(tempimage,tempimage)

# mainimg, modelimg = Image_prepareation(tempimage,(640,640))
# run(mainimg,modelimg)
# cv2.imshow('image', tempimage)
# cv2.waitKey(1000)
