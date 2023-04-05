from dataset import *
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



batch_size_train = 128
batch_size_test = 1
img_size = 256
lr = 0.0001


train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((img_size + 30, img_size+30)),
            transforms.RandomCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
test_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

# define the dataset & transfer to dataloader
trainA = ImageFolder(os.path.join('selfie2anime', 'trainA'), train_transform)
trainB = ImageFolder(os.path.join('selfie2anime', 'trainB'), train_transform)
testA = ImageFolder(os.path.join('selfie2anime', 'testA'), test_transform)
testB = ImageFolder(os.path.join('selfie2anime', 'testB'), test_transform)
trainA_loader = DataLoader(trainA, batch_size=batch_size_train, shuffle=True)
trainB_loader = DataLoader(trainB, batch_size=batch_size_train, shuffle=True)
testA_loader = DataLoader(testA, batch_size=batch_size_test, shuffle=False)
testB_loader = DataLoader(testB, batch_size=batch_size_test, shuffle=False)





if __name__ == '__main__':


    gan = UGATIT()

    # import torchvision.transforms as T
    # for data, label in testA_loader:
    #     print(data[0], label)
    #     trans = T.ToPILImage()
    #     img = trans(data[0])
    #     img.show()
    #     break
