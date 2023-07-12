import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch
from vit_pytorch import ViT
import cv2
from PIL import Image


batch_size_train = 128
batch_size_test = 1
img_size = 256
lr = 0.0001





def box_crop(img, crop_size=224):
    h, w, c = img.shape
    x, y = h // 2, w // 2
    img_center = (x, y)
    x1 = x - crop_size // 2
    y1 = y - crop_size // 2
    x2 = x + crop_size // 2
    y2 = y + crop_size // 2
    # return x1, y1, x2, y2
    return img[y1:y2, x1:x2, :]

def img_crop():
    src_folder = '/Users/chenzhuo/Desktop/github/UGATIT/dataset/selfie2anime'
    dest_folder = './data'
    for folder in [i for i in os.listdir(src_folder) if not i.startswith(".")]:
        for img in [i for i in os.listdir(os.path.join(src_folder, folder)) if not i.startswith(".")]:
            print(folder, img, end="  ")
            image = cv2.imread(os.path.join(src_folder, folder, img))
            print(image.shape, end="  ")
            img_crop = box_crop(image)
            print(img_crop.size)
            dest = os.path.join(dest_folder, folder)
            if not os.path.isdir(dest):
                os.makedirs(dest)
            cv2.imwrite(os.path.join(dest, img), img_crop)


def build_dataset():
    trans = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
    # define the dataset & transfer to dataloader
    train = ImageFolder(os.path.join('data', 'train'), trans)
    test = ImageFolder(os.path.join('data', 'test'), trans)
    train_loader = DataLoader(train, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size_train, shuffle=True)
    return train_loader, test_loader


def train():
    pass


if __name__ == '__main__':
    model = ViT(
        image_size=256,
        patch_size=32,
        num_classes=2,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )

    train_loader, test_loader = build_dataset()

    print(train_loader)
    # img_crop()