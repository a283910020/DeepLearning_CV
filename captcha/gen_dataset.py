import string
from captcha.image import ImageCaptcha
import torch
import random
from torchvision.transforms.functional import to_tensor, to_pil_image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image


def img_generator(width, height, characters, train_test):
    generator = ImageCaptcha(width=width, height=height)
    random_str = ''.join([random.choice(characters) for j in range(4)])
    image = to_tensor(generator.generate_image(random_str))
    # print(image.shape)
    transform = T.ToPILImage()
    img = transform(image)
    img.save(f"data/{train_test}/{random_str}.jpg")


if __name__ == '__main__':

    characters = string.digits + string.ascii_uppercase
    width, height, n_len = 192, 64, 44444
    print(characters, width, height, n_len)

    for i in range(10000):
        img_generator(width, height, characters, "train")
    for i in range(1000):
        img_generator(width, height, characters, "valid")

