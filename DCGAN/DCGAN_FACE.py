## written for the DCGAN model to generate fake face image
## using CPU for trainning

import random
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import time


# init weight for D and G  mean=0/1 stdev = 0.02
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(bias_size, G_feature_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(G_feature_size * 8),
            nn.ReLU(inplace=True),
            # state size. (G_feature_size*8) x 4 x 4
            nn.ConvTranspose2d(G_feature_size * 8, G_feature_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_feature_size * 4),
            nn.ReLU(inplace=True),
            # state size. (G_feature_size*4) x 8 x 8
            nn.ConvTranspose2d(G_feature_size * 4, G_feature_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_feature_size * 2),
            nn.ReLU(inplace=True),
            # state size. (G_feature_size*2) x 16 x 16
            nn.ConvTranspose2d(G_feature_size * 2, G_feature_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_feature_size),
            nn.ReLU(inplace=True),
            # state size. (G_feature_size) x 32 x 32
            nn.ConvTranspose2d(G_feature_size, num_channel, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (num_channel) x 64 x 64
        )

    def forward(self, input):
        return self.model(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # input is (num_channel) x 64 x 64
            nn.Conv2d(num_channel, D_feature_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (D_feature_size) x 32 x 32
            nn.Conv2d(D_feature_size, D_feature_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_feature_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (D_feature_size*2) x 16 x 16
            nn.Conv2d(D_feature_size * 2, D_feature_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_feature_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (D_feature_size*4) x 8 x 8
            nn.Conv2d(D_feature_size * 4, D_feature_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_feature_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (D_feature_size*8) x 4 x 4
            nn.Conv2d(D_feature_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)


if __name__ == '__main__':

    manualSeed = random.randint(1, 10000)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    dataroot = "data/img_align_celeba"
    workers = 6
    batch_size = 128
    image_size = 64
    num_channel = 3  # input channel num
    bias_size = 100  # equals to the generator input
    G_feature_size = 64
    D_feature_size = 64
    num_epochs = 1
    lr = 0.0002
    betas0 = 0.5  # betas[0] for Adam optimizer

    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:150])  # too large for full size

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    # device = torch.device("cpu")

    netG = Generator().apply(weights_init)
    netD = Discriminator().apply(weights_init)
    criterion = nn.BCELoss()  # Binary Cross Entropy

    # optimizer for D and G
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(betas0, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(betas0, 0.999))

    fixed_noise = torch.randn(64, bias_size, 1, 1)
    real_label, fake_label = 1, 0
    img_list, G_losses, D_losses, iters = [], [], [], 0

    time_start = time.time()
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            # update D: maximum log(D(x)) + log(1 - D(G(z)))
            netD.zero_grad()
            single_batch_size = data[0].size(0)
            label = torch.full((single_batch_size,), real_label, dtype=torch.float)
            output = netD(data[0]).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(single_batch_size, bias_size, 1, 1)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake
            optimizerD.step()

            # update G: maximum log(D(G(z)))
            netG.zero_grad()  # init gradient param to 0
            label.fill_(real_label)  # fake data are real ones for G
            # D had been updated we just push forward
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if i % 5 == 0:
                time_stop = time.time()
                time_delta = time_stop - time_start
                print(f'time cost {time_delta} [{epoch + 1}/{num_epochs}][{i + 1}/{len(dataloader)}]\tLoss_D: {errD.item()}\tLoss_G: {errG.item()}\tD(x): {D_x}\tD(G(z)): {D_G_z1} / {D_G_z2}')
                time_start = time_stop

            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if (not iters % 500) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    real_batch = next(iter(dataloader))

    # real img
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

    # fake img
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.show()

