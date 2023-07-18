from os.path import dirname, abspath
import sys, os
d = dirname(dirname(abspath(__file__)))
sys.path.append(d)

from torch import optim
import torch.nn.functional as F

from implicitdl import ImplicitModel
from implicitdl.utils import train
from load_data import mnist_load


epochs = 10
bs = 100
lr = 5e-3

train_ds, train_dl, valid_ds, valid_dl = mnist_load(bs)

n = 100 # the main parameter of an implicit model, determining the size of the hidden state matrix X 
p = 784 # the flattened input size, in this case 28 x 28 (pixels) for MNIST
q = 10 # the output size

model = ImplicitModel(n, p, q)
opt = optim.Adam(ImplicitModel.parameters(model), lr=lr)
loss_fn = F.cross_entropy

model, _ = train(model, train_dl, valid_dl, opt, loss_fn, epochs, "MNIST_Implicit_100_Inf")