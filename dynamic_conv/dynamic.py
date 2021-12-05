# %%
"""
# Training ResNets on CIFAR-10
In this project, you will use the ResNets model to perform image classification on CIFAR-10. CIFAR-10 orginally contains 60K images from 10 categories. We split it into 45K/5K/10K images to serve as train/valiation/test set. We only release the ground-truth labels of training/validation dataset to you.
"""

# %%
"""
## Step 0: Set up the ResNets model
As you have practiced to implement simple neural networks in Homework 1, we just prepare the implementation for you.
"""

# %%
# import necessary dependencies
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
matplotlib.use('Agg')
import math
import model
# %%
# useful libraries
import torchvision
import torchvision.transforms as transforms
def write_output():
    f = open("result/%s.csv"%(sys.argv[1]),'w')
    f.write("training,val\n")
    for i in range(len(print_total_loss[0])):
        f.write("%f,%f\n"%(print_total_loss[0][i],print_total_loss[1][i]))
    plt.plot(print_total_loss[0],c='r',label='training')
    plt.ylabel("accuracy")
    plt.ylim(0,1)
    plt.xlabel("# of epochs")
    plt.title("kernel size = %s"%(sys.argv[1]))
    plt.plot(print_total_loss[1],c='b',label = 'val')
    plt.legend(loc="lower right")
    plt.savefig("result/%s.png"%(sys.argv[1]))
#############################################
# your code here
# specify preprocessing function
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=(32,32),padding = 4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
#############################################

# %%
# do NOT change these
from tools.dataset import CIFAR10
from torch.utils.data import DataLoader

# a few arguments, do NOT change these
DATA_ROOT = "./data"
TRAIN_BATCH_SIZE = 256
VAL_BATCH_SIZE = 100

#############################################
# your code here
# construct dataset
train_set = CIFAR10(
    root=DATA_ROOT, 
    mode='train', 
    download=True,
    transform=transform_train    # your code
)
val_set = CIFAR10(
    root=DATA_ROOT, 
    mode='val', 
    download=True,
    transform=transform_val     # your code
)
test_set = CIFAR10(
    root=DATA_ROOT, 
    mode='test', 
    download=True,
    transform=transform_val
)
# construct dataloader
train_loader = DataLoader(
    train_set, 
    batch_size=TRAIN_BATCH_SIZE,  # your code
    shuffle=True,     # your code
    num_workers=4
)
val_loader = DataLoader(
    val_set, 
    batch_size=VAL_BATCH_SIZE,  # your code
    shuffle= True,     # your code
    num_workers=4
)
test_loader = DataLoader(
    test_set, 
    batch_size=TRAIN_BATCH_SIZE, 
    shuffle=False, 
    num_workers=4)
#############################################
class WarmUP(_LRScheduler):
    def __init__(self, optimizer, lr, num_epo=30, last_epoch=-1):
        self.lr = lr
        self.num_epo = num_epo
        super(WarmUP, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch % self.num_epo == 0 and self.last_epoch != 0:
            self.lr = self.lr / 10
        return [self.lr]
#############################################
# your code here
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Model Definition  
print("running with kernel size %s"%(sys.argv[1]))
net = ResNets(int(sys.argv[1]))
net = net.to(device)
print('Using device:', device)
# nvidia-smi
#############################################
# %%
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
# hyperparameters, do NOT change right now
INITIAL_LR = 0.1
EPOCHS = 100
MOMENTUM = 0.9
REG = 1e-4
criterion = nn.CrossEntropyLoss()

# Add optimizer
optimizer = optim.SGD(net.parameters(), lr=INITIAL_LR, momentum=MOMENTUM,weight_decay=REG,nesterov=True)
# scheduler = ReduceLROnPlateau(optimizer, factor=0.1,threshold = 1e-4,patience=8, verbose=True)
scheduler = WarmUP(optimizer,INITIAL_LR,30)
#############################################


CHECKPOINT_PATH = "./save_model"

# start the training/validation process
# the process should take about 5 minutes on a GTX 1070-Ti
# if the code is written efficiently.
best_val_acc = 0
current_learning_rate = INITIAL_LR
print_total_loss = [[],[]]
print("==> Training starts!")
print("="*50)
for i in range(0, EPOCHS):
    # handle the learning rate scheduler.
    
    net.train()
    print("\nEpoch [{}/{}]".format(i+1, EPOCHS))
    print("learning rate: ",optimizer.param_groups[0]['lr'])
    current_learning_rate = optimizer.param_groups[0]['lr']
    
    #######################
    
    print("Epoch %d:" %(i+1))
    # this help you compute the training accuracy
    total_examples = 0
    correct_examples = 0

    train_loss = 0 # track training loss if you want
    
    # Train the model for 1 epoch.
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        ####################################
        # your code here
        # copy inputs to device
        inputs = inputs.to(device)
        targets = targets.to(device)
        out = net(inputs)
        loss = criterion(out,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _,predicted = torch.max(out, 1)
        correct = predicted.eq(targets).sum()
        train_loss+=loss
        total_examples+=targets.shape[0]
        correct_examples += correct.item()
        ####################################
    
    avg_loss = train_loss / len(train_loader)
    train_avg_acc = correct_examples / total_examples
    print("Training loss: %.4f, Training accuracy: %.4f" %(avg_loss, train_avg_acc))
    net.update_temp(i+1)
#     print(train_avg_acc)

    # Validate on the validation dataset
    #######################
    # your code here
    # switch to eval mode
    scheduler.step()
    net.eval()
    
    
    #######################

    # this help you compute the validation accuracy
    total_examples = 0
    correct_examples = 0
    
    val_loss = 0 # again, track the validation loss if you want

    # disable gradient during validation, which can save GPU memory
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            ####################################
            # your code here
            # copy inputs to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            out = net(inputs)
            loss = criterion(out,targets)
            _,predicted = torch.max(out, 1)
            correct = predicted.eq(targets).sum()
            val_loss+=loss
            total_examples+=targets.shape[0]
            correct_examples += correct.item()
            ####################################

    avg_loss = val_loss / len(val_loader)
    avg_acc = correct_examples / total_examples
    print("Validation loss: %.4f, Validation accuracy: %.4f" % (avg_loss, avg_acc))
    print_total_loss[0].append(train_avg_acc)
    print_total_loss[1].append(avg_acc)
    # save the model checkpoint
    if avg_acc > best_val_acc:
        best_val_acc = avg_acc
        if not os.path.exists(CHECKPOINT_PATH):
            os.makedirs(CHECKPOINT_PATH)
        print("Saving ...")
        state = {'state_dict': net.state_dict(),
                 'epoch': i,
                 'lr': current_learning_rate}
        # torch.save(state, os.path.join(CHECKPOINT_PATH, 'ResNets_%s.pth'%(sys.argv[1])))
        
    print('')

print("="*50)
print(f"==> Optimization finished! Best validation accuracy: {best_val_acc:.4f}")
# write_output()
