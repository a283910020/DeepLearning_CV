import torch
from vit_pytorch import ViT

def test():
    v = ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 2,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    img = torch.randn(10, 3, 256, 256)

    preds = v(img)
    print(preds)
    # assert preds.shape == (100, 20), 'correct logits outputted'
    # print("test pass")

def img_crop(img_folder):
    pass

if __name__ == '__main__':
    test()