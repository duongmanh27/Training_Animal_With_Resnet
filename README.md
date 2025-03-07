Transfer Learning with ResNet-50 on Pascal VOC Animal Dataset
Introduction
This project demonstrates how to use transfer learning to train a convolutional neural network (CNN) based on the ResNet-50
architecture for classifying animal images from the Pascal VOC dataset. Transfer learning leverages the pre-trained weights of 
a deep neural network that was trained on a large dataset (e.g., ImageNet) and fine-tunes it on a specific target dataset, 
significantly reducing the amount of data and time required for effective training.

During the training process, the loss starts at a high level, decreases and then stabilizes with periodic sharp peaks.

![Train Loss](images_tensorboard/train_loss.png)

Accuracy: Increased from 0.88 to 0.95 in 100 epochs.

Loss: Reduced from 0.34 to 0.18 in 100 epochs
![Val Accuracy_Loss](images_tensorboard/val_acc_loss.png)
