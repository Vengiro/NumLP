import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.detection.faster_rcnn as fasterrcnn
import ultralytics.models.yolo.model as yolo
import json
import os
from PIL import Image
import torch
from torchvision import transforms
import time

confidence_treshold = 0.5
iou_treshold = 0.5
path_image = "val2017"
path_ann = "annotations/instances_val2017.json"
# Mapping from annotations class to models class because they are a discrepancy between the two.......
NINTY_TO_EIGHTY = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "11": 11, "13": 12, "14": 13, "15": 14, "16": 15, "17": 16, "18": 17, "19": 18, "20": 19, "21": 20, "22": 21, "23": 22, "24": 23, "25": 24, "27": 25, "28": 26, "31": 27, "32": 28, "33": 29, "34": 30, "35": 31, "36": 32, "37": 33, "38": 34, "39": 35, "40": 36, "41": 37, "42": 38, "43": 39, "44": 40, "46": 41, "47": 42, "48": 43, "49": 44, "50": 45, "51": 46, "52": 47, "53": 48, "54": 49, "55": 50, "56": 51, "57": 52, "58": 53, "59": 54, "60": 55, "61": 56, "62": 57, "63": 58, "64": 59, "65": 60, "67": 61, "70": 62, "72": 63, "73": 64, "74": 65, "75": 66, "76": 67, "77": 68, "78": 69, "79": 70, "80": 71, "81": 72, "82": 73, "84": 74, "85": 75, "86": 76, "87": 77, "88": 78, "89": 79, "90": 80}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def IoU(box1, box2):


    x1min, y1min, x1max, y1max = box1
    x2min, y2min, x2max, y2max = box2

    xA = max(x1min, x2min)
    yA = max(y1min, y2min)
    xB = min(x1max, x2max)
    yB = min(y1max, y2max)
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (x1max - x1min) * (y1max - y1min)
    box2Area = (x2max - x2min) * (y2max - y2min)
    iou = interArea / float(box1Area + box2Area - interArea)
    return iou

def prAndrcl(prediction, truth, nb_truth):

    # Sort predictions by confidence
    prediction = sorted(prediction, key=lambda x: x[1], reverse=True)
    tp = 0
    fp = 0
    fn = nb_truth
    matched_true_ind = set()
    precision = []
    recall = []
    for pred in prediction:
        if pred[3] not in truth:
            continue
        real_res = truth[pred[3]]
        best_iou = 0
        current_iou = 0
        best_iou_ind = None
        for ind, true in enumerate(real_res):
            #Find best match
            if int(pred[2])==(NINTY_TO_EIGHTY[str(true[1])]-1) and ind not in matched_true_ind:
                current_iou = IoU(pred[0], true[0])
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_iou_ind = ind

        if best_iou > iou_treshold:
            tp += 1
            matched_true_ind.add(best_iou_ind)
            fn -= 1
        else:
            fp += 1

        prc = tp/(tp+fp) if tp+fp>0 else 0
        rcl = tp/(tp+fn) if tp+fn>0 else 0
        precision.append(prc)
        recall.append(rcl)

    return precision, recall

"""
Calculate Average Precision using discrete integration
"""
def AP(precision, recall):

    ap = 0
    for i in range(1, len(precision)):
        ap += precision[i] * (recall[i] - recall[i-1])
    ap/= (len(precision)-1) if len(precision)>1 else 1
    return ap
def mAP(predictions_class, truths, classes, nb_per_class):

    aps = []
    for cls in classes:
        precision = []
        recall = []

        # If there is less than two pred for this class, we can't calculate the integral
        if len(predictions_class[cls])>1:
            precision, recall = prAndrcl(predictions_class[cls], truths, nb_per_class[cls])
            ap = AP(precision, recall)
            aps.append(ap)


    return np.mean(aps)



def main():
    # Load the model
    faster_rcnn = fasterrcnn.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
    faster_rcnn.eval()

    yolo_model = yolo.YOLO("yolov8s.pt").to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    predictions_yolo = []
    predictions_frcnn = []
    t_yolo = 0
    t_rcnn = 0
    iteration = 0
    # Cry in preprocessing.......

    for image_name in os.listdir(path_image):
        if iteration == 1:
            break
        # Load image
        img_path = os.path.join(path_image, image_name)
        img = Image.open(img_path)
        img = img.convert("RGB")
        # Need to pad the image since YOLO want multiple of 32
        img = np.array(img)
        pad_y = [[[0, 0, 0] for i in range  (img.shape[1])] for j in range(32 - img.shape[0] % 32)]
        img = np.concatenate((img, pad_y), axis=0)
        pad_x = [[[0, 0, 0] for i in range(32 - img.shape[1] % 32)] for j in range(img.shape[0])]
        img = np.concatenate((img, pad_x), axis=1)
        img = transform(img).unsqueeze(0).float()
        img/=255
        img = img.to(device)
        id_img = image_name.split(".")[0]
        id_img = int(id_img)


        # Make prediction
        with torch.no_grad():
            st = time.time()
            pred_y = yolo_model(img,verbose=False)
            t_yolo += time.time() - st
            cls = pred_y[0].boxes.cls.cpu().numpy()
            conf = pred_y[0].boxes.conf.cpu().numpy()
            boxes = pred_y[0].boxes.xyxy.cpu().numpy()
            for i in range(len(boxes)):
                if conf[i] > confidence_treshold:
                    predictions_yolo.append([boxes[i], conf[i], cls[i], id_img])



            st = time.time()
            pred_f = faster_rcnn(img)
            t_rcnn += time.time() - st
            boxes = pred_f[0]["boxes"].cpu().numpy()
            scores = pred_f[0]["scores"].cpu().numpy()
            label = pred_f[0]["labels"].cpu().numpy()
            for i in range(len(boxes)):
                if scores[i] > confidence_treshold:
                    predictions_frcnn.append([boxes[i], scores[i], label[i], id_img])


        iteration += 1

    t_yolo /= iteration
    t_rcnn /= iteration


    # Load annotations
    with open(path_ann) as f:
        data = json.load(f)
    truths = {}
    for ann in data["annotations"]:
        boxes = ann["bbox"]
        boxes = [boxes[0], boxes[1], boxes[0]+boxes[2], boxes[1]+boxes[3]]
        label = ann["category_id"]
        res = [boxes, label]
        if ann["image_id"] in truths:
            truths[ann["image_id"]].append(res)
        else:
            truths[ann["image_id"]] = [res]

    pred_yolo_class = {}
    pred_frcnn_class = {}

    # Sort predictions by class
    classes= set()
    nb_per_class = {}
    for img in truths:
        for box, cls in truths[img]:
            classes.add(NINTY_TO_EIGHTY[str(cls)]-1)
            if NINTY_TO_EIGHTY[str(cls)]-1 in nb_per_class:
                nb_per_class[NINTY_TO_EIGHTY[str(cls)]-1] += 1
            else:
                nb_per_class[NINTY_TO_EIGHTY[str(cls)]-1] = 1

    for cls in classes:
        pred_yolo_class[cls] = []
        pred_frcnn_class[cls] = []
        for pred in predictions_yolo:
            if pred[2] == cls:
                pred_yolo_class[cls].append(pred)
        for pred in predictions_frcnn:
            if pred[2] == cls:
                pred_frcnn_class[cls].append(pred)


    mAP_yolo = mAP(pred_yolo_class, truths, classes, nb_per_class)
    mAP_frcnn = mAP(pred_frcnn_class, truths, classes, nb_per_class)

    print(f"The latency of YOLO is {t_yolo} and the mAP is {mAP_yolo}")
    print(f"The latency of FasterRCNN is {t_rcnn} and the mAP is {mAP_frcnn}")


if __name__ == "__main__":
    main()