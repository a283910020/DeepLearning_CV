import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

embedding = nn.Embedding(10, 128)
transformer = nn.Transformer(d_model=128, batch_first=True)

src = torch.LongTensor([1,2,3,4,5,6,7,8,9,0])
tgt = torch.LongTensor([1,2,3,4,5,6,7,8,9])

output = transformer(embedding(src), embedding(tgt))

print(output)
