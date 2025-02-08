import cv2 as cv
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize, Compose, ToTensor


class DatasetCifar(Dataset) :

    def __init__(self, path, is_train, transform=None) :
        self.classes = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]
        self.list_images = []
        self.list_label = []
        if is_train :
            data_file = os.path.join(path, "train")
        else :
            data_file = os.path.join(path, "test")
        for idx, file_name in enumerate(os.listdir(data_file)) :
            data_path = os.path.join(data_file, file_name)

            for img_name in os.listdir(data_path) :
                image_path = os.path.join(data_path, img_name)
                self.list_images.append(image_path)
                self.list_label.append(idx)
        self.transform = transform

    def __len__(self) :
        return len(self.list_label)

    def __getitem__(self, index) :
        image = cv.imread(self.list_images[index])
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # image = Image.open(self.image_paths[index]).convert("RGB")
        if self.transform :
            image = self.transform(image)
        label = self.list_label[index]

        return image, label


if __name__ == '__main__' :
    transform = Compose([
        ToTensor(),
        Resize((224, 224))
    ])
    dataset = DatasetCifar("E:\\Code\\Python\\Data\\Data_DL\\animals_v2\\animals", is_train=False, transform=transform)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=16,
        shuffle=True,
        drop_last=True,
        num_workers=2,
    )
    for image, label in dataloader :
        print(image.shape, label.shape)
