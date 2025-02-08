import os.path
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, ToTensor, Resize, RandomRotation, RandomHorizontalFlip
import torch.nn as nn
import torch
from torchvision.models import resnet50, ResNet50_Weights
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from dataset_animal import DatasetCifar


def get_args() :
    parser = argparse.ArgumentParser(description="Animal classifier")
    parser.add_argument("--batch_size", "-b", type=int, default=16, help="batch size of training process")
    parser.add_argument("--image_size", "-i", type=int, default=224, help="Common size of all images")
    parser.add_argument("--data_path", "-d", type=str, default="E:\\Code\\Python\\Data\\Data_DL\\animals_v2\\animals",
        help="Path to data")
    parser.add_argument("--epochs", "-e", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", "-l", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--tensorboard-dir", "-t", type=str, default="animal_board",
        help="where to store the tensorboard logging")
    parser.add_argument("--checkpoint-dir", "-c", type=str, default="trained_models",
        help="where to store the checkpoint")
    parser.add_argument("--resume", "-r", type=bool, default=False, help="Continue training from last session")
    args, unknown = parser.parse_known_args()
    return args


def plot_confusion_matrix(writer, cm, class_names, epoch) :
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="Wistia")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]) :
        for j in range(cm.shape[1]) :
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('Confusion_matrix', figure, epoch)


def train(args) :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform_train = Compose([
        ToTensor(),
        Resize((args.image_size, args.image_size)),
        RandomHorizontalFlip(),
        RandomRotation(degrees=15)
    ])
    transform_val = Compose([
        ToTensor(),
        Resize((args.image_size, args.image_size)),
    ])
    train_dataset = DatasetCifar(path=args.data_path, is_train=True, transform=transform_train)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        drop_last=True
    )
    val_dataset = DatasetCifar(path=args.data_path, is_train=False, transform=transform_val)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        drop_last=False
    )

    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=10)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    if args.resume :
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, "last.pt"))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
    else :
        start_epoch = 0
        best_acc = -1

    if not os.path.isdir(args.tensorboard_dir) :
        os.makedirs(args.tensorboard_dir)
    writer = SummaryWriter(args.tensorboard_dir)
    if not os.path.isdir(args.checkpoint_dir) :
        os.makedirs(args.checkpoint_dir)
    num_iters_per_epoch = len(train_dataloader)
    for epoch in range(start_epoch, args.epochs) :
        # TRAINING STAGE
        model.train()
        progress_bar = tqdm(train_dataloader, colour="white")
        losses = []
        for iter, (images, labels) in enumerate(progress_bar) :
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            avg_loss = np.mean(losses)
            progress_bar.set_description(
                "Epoch: {}/{}. Loss: {:0.4f}".format(epoch + 1, args.epochs, avg_loss))
            writer.add_scalar(tag="Train/Loss", scalar_value=avg_loss, global_step=epoch * num_iters_per_epoch + iter)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # VALIDATION STAGE
        model.eval()
        losses = []
        all_labels = []
        all_predictions = []
        progress_bar = tqdm(val_dataloader, colour="yellow")
        for iter, (images, labels) in enumerate(progress_bar) :
            model.zero_grad()
            all_labels.extend(labels)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.tolist())
            loss = criterion(outputs, labels)
            losses.append(loss.item())
        avg_loss = np.mean(losses)
        avg_acc = accuracy_score(all_labels, all_predictions)
        print("Epoch: {}/{}. Loss: {:0.4f}. Accuracy: {:0.4f}".format(epoch + 1, args.epochs, avg_loss,
            avg_acc))
        writer.add_scalar(tag="Val/Loss", scalar_value=avg_loss, global_step=epoch)
        writer.add_scalar(tag="Val/Accuracy", scalar_value=avg_acc, global_step=epoch)
        plot_confusion_matrix(writer, confusion_matrix(all_labels, all_predictions), train_dataset.classes, epoch)

        checkpoint = {
            "epoch" : epoch + 1,
            "model" : model.state_dict(),
            "optimizer" : optimizer.state_dict(),
            "best_acc" : best_acc
        }

        torch.save(checkpoint, os.path.join(args.checkpoint_dir, "last.pt"))
        if avg_acc > best_acc :
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, "best.pt"))
            best_acc = avg_acc
        scheduler.step(avg_loss)


if __name__ == '__main__' :
    args = get_args()
    train(args)
