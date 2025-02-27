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

def IoU(box1, box2, yolo=True):

    if yolo:
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        x1min, y1max, x1max, y1max = x1-w1/2, y1-h1/2, x1+w1/2, y1+h1/2
        x2min, y2max, x2max, y2max = x2-w2/2, y2-h2/2, x2+w2/2, y2+h2/2
    else:
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

def prAndrcl(prediction, truth, model=True):

    # Sort predictions by confidence
    prediction = sorted(prediction, key=lambda x: x[1], reverse=True)
    tp = 0
    fp = 0
    fn = len(truth)
    matched_true_ind = set()
    precision = []
    recall = []
    for pred in prediction:
        best_iou = 0
        current_iou = 0
        best_iou_ind = None
        for ind, true in enumerate(truth):
            #Find best match
            if pred[2]==true[2] and ind not in matched_true_ind:
                current_iou = IoU(pred[0], true[0], yolo=model)
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
    ap/= (len(precision)-1)
    return ap
def mAP(predictions_class, truths_class, model=True):

    aps = []
    for class_pred, class_truth in zip(predictions_class, truths_class):
        precision, recall = prAndrcl(class_pred, class_truth, model)
        aps.append(AP(precision, recall))

    return np.mean(aps)



def main():
    # Load the model
    faster_rcnn = fasterrcnn.fasterrcnn_resnet50_fpn(pretrained=True)
    faster_rcnn.eval()

    yolo_model = yolo.YOLO("yolov8s.pt")

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    predictions_yolo = {}
    predictions_frcnn = {}
    t_yolo = 0
    t_rcnn = 0
    iteration = 0
    for image_name in os.listdir(path_image):
        # Load image
        img_path = os.path.join(path_image, image_name)
        img = Image.open(img_path)
        # Need to pad the image since YOLO want multiple of 32
        img = np.array(img)
        pad_y = [[[0, 0, 0] for i in range  (img.shape[1])] for j in range(32 - img.shape[0] % 32)]
        img = np.concatenate((img, pad_y), axis=0)
        pad_x = [[[0, 0, 0] for i in range(32 - img.shape[1] % 32)] for j in range(img.shape[0])]
        img = np.concatenate((img, pad_x), axis=1)
        img = transform(img).unsqueeze(0).float()
        img/=255



        # Make prediction
        with torch.no_grad():  # No gradient computation during inference
            st = time.time()
            pred_y = yolo_model(img)
            t_yolo += time.time()-st
            st = time.time()
            pred_f = faster_rcnn(img)
            t_rcnn += time.time()-st

        if predictions_yolo.get(image_name) is None:
            predictions_yolo[image_name] = [pred_y]
        else:
            predictions_yolo[image_name].append(pred_y)

        if predictions_frcnn.get(image_name) is None:
            predictions_frcnn[image_name] = [pred_f]
        else:
            predictions_frcnn[image_name].append(pred_f)
        iteration += 1

    t_yolo /= iteration
    t_rcnn /= iteration


    # Load annotations
    with open(path_ann) as f:
        data = json.load(f)
    truths = {}
    for ann in data["annotations"]:
        if ann["image_id"] in truths:
            truths[ann["image_id"]].append(ann)
        else:
            truths[ann["image_id"]] = [ann]

    mAP_yolo = mAP(predictions_yolo, truths, model=True)
    mAP_frcnn = mAP(predictions_frcnn, truths, model=False)

    print(f"The latency of YOLO is {t_yolo} and the mAP is {mAP_yolo}")
    print(f"The latency of FasterRCNN is {t_rcnn} and the mAP is {mAP_frcnn}")


if __name__ == "__main__":
    main()