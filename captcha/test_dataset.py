from captcha_reco import CaptchaDataset
import string
from matplotlib import pyplot as plt

characters = '-' + string.digits + string.ascii_uppercase
width, height, n_len, n_classes = 192, 64, 4, len(characters)
train_set = CaptchaDataset(characters, 10000, width, height, 12, n_len, "data/train")

plt.figure(figsize=(8,8))
for i in range(9):
    img, target, _, _ = train_set[i]
    img = img.permute(1,2,0)
    ax=plt.subplot(3,3,i+1)
    ax.imshow(img.numpy())
    ax.set_title(f"{target}")
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()