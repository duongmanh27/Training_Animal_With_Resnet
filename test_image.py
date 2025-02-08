import torch
import torch.nn as nn
import numpy as np
import os
import cv2 as cv
import warnings
import argparse
from torchvision.models import ResNet50_Weights, resnet50


bufterfly = "E:\\Code\\Python\\Deep_Learning\\Training_Animal_With_Resnet50\\test_image_bufterfly.jpg"


def get_args() :
    parser = argparse.ArgumentParser(description="Animal classifier")
    parser.add_argument("--image_path", "-p", type=str, default=bufterfly, help="Input image")
    parser.add_argument("--image_size", "-i", type=int, default=224, help="Common size of all images")
    parser.add_argument("--checkpoint-dir", "-c", type=str, default="trained_models",
        help="where to store the checkpoint")
    args = parser.parse_args()
    return args


def inference(args) :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=10)
    checkpoint = torch.load(os.path.join(args.checkpoint_dir, "best.pt"))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    ori_image = cv.imread(args.image_path)
    image = cv.cvtColor(ori_image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, (args.image_size, args.image_size))
    image = image / 255.
    image = np.transpose(image, (2, 0, 1))[None, :, :, :]
    image = torch.from_numpy(image)
    softmax = nn.Softmax()
    classes = ["butterfly", "cat", "chicken", "cow", "dog",
               "elephant", "horse", "sheep", "spider", "squirrel"]
    with torch.no_grad() :
        image = image.float().to(device)
    output = model(image)
    prob = softmax(output)[0]
    predicted_class = classes[torch.argmax(prob)]
    cv.imshow("Prediction: {} ({:0.2f}%)"
    .format(predicted_class, torch.max(prob) * 100), ori_image)
    cv.waitKey(0)


if __name__ == '__main__' :
    args = get_args()
    inference(args)
