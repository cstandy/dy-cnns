from tools.dataset import CIFAR10
from torch.utils.data import DataLoader
import os, sys
import time
import datetime
from tqdm import tqdm_notebook as tqdm
from torch.optim.lr_scheduler import _LRScheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
from model import ResNets
from model import DynamicConv
import math
import model
import torchvision
import torchvision.transforms as transforms
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
DATA_ROOT = "./data"
TRAIN_BATCH_SIZE = 256
VAL_BATCH_SIZE = 100
class_name = [ 'deer','horse','frog','truck','airplane','cat','dog','ship','bird','automobile']
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
def plot_pic(value,target,K,path):
    plt.close()
    ax = plt.subplot()
    im = ax.imshow(value)
    # deer frog horse dog ? ? boat cat frog cat ? truck bird ? ?
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    labels= []
    loc = []
    for i in range(20):
        labels.append(i+1)
        loc.append(i)
    # set the xticks location
    ax.set_xticks(loc)
    # set the value
    ax.set_xticklabels(labels)
    ax.set_yticks([i for i in range(K)])
    ax.set_yticklabels([i+1 for i in range(K)])
    ax.set_xlabel("block number",loc = "center")
    ax.set_ylabel("attention scores",loc = "center")
    fig = plt.gcf()
    fig.set_size_inches((8.5, 11), forward=False)
    plt.title("attention scores on %s"%(class_name[target]))
    fig.savefig("%s%d_num_%s.png"%(path,K,class_name[target]),dpi=500)
    plt.colorbar(im, cax=cax)
    plt.show()
def plot_attention(inputs,device,target,net,K,path):
    inputs = inputs.to(device)
    out = net(inputs)
    attention_layer=[]
    for name, module in net.named_modules():
        if isinstance(module, DynamicConv):
            attention = module.attention_score.cpu().detach().numpy()
            attention_layer.append(attention)

    attention_layer = np.array(attention_layer).reshape((-1,K,20))
    for i in range(4):
        plot_pic(attention_layer[i],target[i],K,path)
if __name__ == '__main__':
    # model_path = "saved_model/ResNets.pth"
    model_path = sys.argv[1]
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_set = CIFAR10(
        root=DATA_ROOT,
        mode='test',
        download=True,
        transform=transform_val
    )
    val_set = CIFAR10(
        root=DATA_ROOT,
        mode='val',
        download=True,
        transform=transform_val     # your code
    )
    val_loader = DataLoader(
        val_set,
        batch_size=VAL_BATCH_SIZE,  # your code
        shuffle= False,     # your code
        num_workers=4
    )
    test_loader = DataLoader(
        test_set,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=4)
    for batch_idx, (inputs,target) in enumerate(test_loader):
            if(batch_idx >0):
                break
    criterion = nn.CrossEntropyLoss()
    K = int(sys.argv[1])
    net = ResNets(K)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.load_state_dict(torch.load(model_path)['state_dict'])
    net = net.to(device)
    for batch_idx, (inputs,target) in enumerate(val_loader):
            if(batch_idx >0):
                break
            # inputs = inputs.to(device)
    plot_attention(inputs,device,target,net,K,"./fig/")
