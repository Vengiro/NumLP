import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet34
from torchvision import models
from PIL import Image
import torch
from torchvision import transforms
import time
from PIL import Image, ImageDraw, ImageFont
from torchvision.models import resnet50
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms


def RPN(image, model, window_sizes, conf_threshold=0.5):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match the model's input
        transforms.ToTensor(),
        # Number from ResNet documentation, apparently if we use the same normalization as the model was trained on, we get better results
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


    if image.mode != "RGB":
        image = image.convert("RGB")  # Ensure image is in RGB format

    model.eval()
    detected_windows = []

    for window_size in window_sizes:
        stride = int(window_size * 0.5)  # 50% overlap
        windows = sliding_window(image, window_size, stride)

        for box, window in windows:
            input_tensor = transform(window).unsqueeze(0)

            with torch.no_grad():
                output = model(input_tensor)
                probabilities = F.softmax(output, dim=1)
                max_prob, predicted = torch.max(probabilities, 1)
                max_prob = max_prob.item()

            if max_prob > conf_threshold:
                detected_windows.append((box, predicted.item(), max_prob))

    return detected_windows


def sliding_window(image, window_size, stride):

    windows = []
    width, height = image.size
    for y in range(0, height - window_size + 1, stride):
        for x in range(0, width - window_size + 1, stride):
            box = (x, y, x + window_size, y + window_size)
            window = image.crop(box)
            windows.append((box, window))
    return windows


def draw_boxes(image, detections):

    draw = ImageDraw.Draw(image)
    for box, pred, conf in detections:
        draw.rectangle(box, outline="red", width=2)
        text = f"{pred} {conf:.2f}"
        draw.text((box[0], box[1]), text, fill="yellow")
    return image

def main():

    #load image
    image_paths = ["2007_001239.jpg", "2008_002152.jpg"]
    images = [Image.open(path) for path in image_paths]


    model = resnet50(pretrained=True)

    window_sizes = [64, 128, 256]
    for i, img in enumerate(images):
        detections = RPN(img, model, window_sizes)

        print(f"Detections in Image {i + 1}:")
        for box, pred, conf in detections:
            print(f"Box: {box}, Predicted class: {pred}, Confidence: {conf:.2f}")

        img_boxed = img.copy()
        img_boxed = draw_boxes(img_boxed, detections)
        img_boxed.show(title=f"Image {i + 1} Detections")


if __name__ == "__main__":
    main()


