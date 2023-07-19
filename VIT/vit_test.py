import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.optim as optim
from torch import nn
import torch
from vit_pytorch import ViT
from torchvision.models.vision_transformer import vit_b_16 as vit_model

import cv2
from PIL import Image

batch_size_train = 128
batch_size_test = 1
img_size = 256
lr = 0.0001


def box_crop(img, crop_size=256):
    h, w, c = img.shape
    x, y = h // 2, w // 2
    img_center = (x, y)
    x1 = x - crop_size // 2
    y1 = y - crop_size // 2
    x2 = x + crop_size // 2
    y2 = y + crop_size // 2
    # return x1, y1, x2, y2
    return img[y1:y2, x1:x2, :]


def image_crop():
    src_folder = '/Users/chenzhuo/Desktop/github/UGATIT/dataset/selfie2anime'
    dest_folder = './data'
    for folder in [i for i in os.listdir(src_folder) if not i.startswith(".")]:
        for img in [i for i in os.listdir(os.path.join(src_folder, folder)) if not i.startswith(".")]:
            print(folder, img, end="  ")
            image = cv2.imread(os.path.join(src_folder, folder, img))
            print(image.shape, end="  ")
            img_crop = box_crop(image, 224)
            print(img_crop.shape)
            dest = os.path.join(dest_folder, folder[:-1], folder)
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
    train_folder = ImageFolder(os.path.join('data', 'train'), trans)
    test_folder = ImageFolder(os.path.join('data', 'test'), trans)
    train_loader = DataLoader(train_folder, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(test_folder, batch_size=batch_size_train, shuffle=True)
    return train_loader, test_loader


def train(model, epoch, criterion, data_loader, device, optimizer):
    train_loss = 0
    model.train()
    criterion = criterion.to(device)
    batch_num = 1
    for data, label in data_loader:
        # print(len(data), len(data_loader))
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
        train_loss = train_loss / len(data_loader)
        print('Epoch: {} \t batch_num: {}/{} \tTraining Loss: {:.6f}'.format(epoch, batch_num, len(data_loader),
                                                                             train_loss))
        batch_num += 1


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # if batch % 100 == 0:
        loss, current = loss.item(), (batch + 1) * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    image_crop()

    device = torch.device("mps")

    batch_size = 256
    num_workers = 4
    lr = 1e-4
    epochs = 1

    # model = ViT(
    #     image_size=256,
    #     patch_size=32,
    #     num_classes=2,
    #     dim=1024,
    #     depth=6,
    #     heads=16,
    #     mlp_dim=2048,
    #     dropout=0.1,
    #     emb_dropout=0.1
    # )

    model = vit_model(pretrained=True)
    print(model)
    in_feature = model.heads.head.in_features
    print(in_feature)
    # fc_in_feature = model.heads.in_features
    model.heads.head = nn.Linear(in_features=in_feature, out_features=2)
    print(model)
    # exit(0)
    model = model.to(device)
    train_loader, test_loader = build_dataset()
    print(len(train_loader), len(test_loader))
    optimizer = optim.Adam(model.parameters(), lr=lr, )
    loss_fn = nn.CrossEntropyLoss()

    # print(train_loader)

    for epoch in range(1, epochs + 1):
        # train(model, epoch, criterion, train_loader, device, optimizer)
        # val(epoch)
        train_loop(train_loader, model, loss_fn, optimizer)
        test_loop(test_loader, model, loss_fn)

#
#
# device = torch.device('mps')
# run_on_gpu = True
# # if torch.cuda.is_available():
# #     device = torch.device('mps')
# #     run_on_gpu = True
#
# x = torch.randn(2, 3, requires_grad=True)
# y = torch.rand(2, 3, requires_grad=True)
# z = torch.ones(2, 3, requires_grad=True)
#
# with torch.autograd.profiler.profile(use_cpu=True) as prf:
#     for _ in range(1000):
#         z = (z / x) * y
#
# print(prf.key_averages().table(sort_by='self_cpu_time_total'))
#
# with torch.autograd.profiler.profile(use_cuda=True) as prf:
#     for _ in range(1000):
#         z = (z / x) * y
#
# print(prf.key_averages().table(sort_by='self_cpu_time_total'))
