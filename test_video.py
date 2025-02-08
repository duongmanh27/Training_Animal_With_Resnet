import cv2 as cv
import torch
import torch.nn as nn
import argparse
import os
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import Compose, ToTensor, Resize


video_cat = "E:\\Code\\Python\\Deep_Learning\\TrainingMNIST_RESNET\\video_cat_02.mp4"


def get_args() :
    parser = argparse.ArgumentParser("Argument in Inference Animal")
    parser.add_argument('--video-path', '-i', default=video_cat, type=str, help="Path to test image")
    parser.add_argument("--image-size", '-s', default=224, type=int, help="Size common all images")
    parser.add_argument("--checkpoint-dir", "-c", default="trained_models", type=str,
        help="where save file trained model")
    args = parser.parse_args()
    return args


def inference(args) :
    name_classes = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(os.path.join(args.checkpoint_dir, "best.pt"))
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=10)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    transform = Compose([
        ToTensor(),
        Resize((224, 224))
    ])
    softmax = nn.Softmax()
    cap = cv.VideoCapture(args.video_path)
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    out = cv.VideoWriter("result.mp4", fourcc, int(cap.get(cv.CAP_PROP_FPS)), (width, height))
    font = cv.FONT_HERSHEY_SIMPLEX
    while cap.isOpened() :
        flag, ori_frame = cap.read()
        if not flag :
            break
        w = ori_frame.shape[0]
        frame = transform(ori_frame)
        frame = frame.unsqueeze(0)
        frame = frame.to(device)
        with torch.no_grad() :
            output = model(frame)
        probity = softmax(output)[0]
        predicted_name = name_classes[torch.argmax(output, dim=1)]
        cv.putText(ori_frame, "Label: {}".format(predicted_name), (150, 150), font, 7,
            (0, 0, 0), 3, cv.LINE_4)
        cv.putText(ori_frame, "Confident score: {:.4f}".format(torch.max(probity) * 100),
            (150, w - 100), font, 7, (0, 0, 0), 3, cv.LINE_4)
        out.write(ori_frame)

    cap.release()
    out.release()


if __name__ == '__main__' :
    args = get_args()
    inference(args)
