from numpy.core.records import array
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
import plot_attention as att
import model
import torch.nn as nn
import hw2
import numpy as np
import torch.optim as optim
import attacks
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
import torchvision.transforms as transforms
import numpy as np
class WarmUP(_LRScheduler):
    def __init__(self, optimizer, lr, num_epo=30, last_epoch=-1):
        self.lr = lr
        self.num_epo = num_epo
        super(WarmUP, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch % self.num_epo == 0 and self.last_epoch != 0:
            self.lr = self.lr / 10
        return [self.lr]
def test_model(mdl, loader, device):
    mdl.eval()
    running_correct = 0.
    running_loss = 0.
    running_total = 0.
    with torch.no_grad():
        for batch_idx,(data,labels) in enumerate(loader):
            data = data.to(device); labels = labels.to(device)
            clean_outputs = mdl(data)
            clean_loss = F.cross_entropy(clean_outputs, labels)
            _,clean_preds = clean_outputs.max(1)
            running_correct += clean_preds.eq(labels).sum().item()
            running_loss += clean_loss.item()
            running_total += labels.size(0)
    clean_acc = running_correct/running_total
    clean_loss = running_loss/len(loader)
    mdl.train()
    return clean_acc,clean_loss
def predict(whitebox,val_loader,device):
    whitebox = ResNets(kernel_num)
    whitebox.load_state_dict(torch.load('./saved_model/ResNets.pth')['state_dict'])
    # whitebox.load_state_dict(torch.load('pretrained/adversarial.pt')['state_dict'])
    whitebox = whitebox.to(device)
    whitebox.eval(); 
    test_acc,_ = test_model(whitebox,val_loader,device)
    print("Initial Accuracy of Whitebox Model: ",test_acc)
    # exit()
    eps_list = np.linspace(0,0.1,11)
    # eps_list = [0.05]
    white_list = []

    for ATK_EPS in eps_list:
        whitebox_correct = 0.
        running_total = 0.
        for batch_idx,(data,labels) in enumerate(val_loader):
            data = data.to(device) 
            labels = labels.to(device)
                
            # adv_data = attacks.random_noise_attack(whitebox,device,data.clone().detach(),ATK_EPS)
            ITS=10
            ALP = 1.85*(ATK_EPS/ITS) 
            white_adv_data = attacks.PGD_attack(whitebox,device,data.clone().detach(),labels,eps=ATK_EPS,alpha=ALP,iters=ITS,rand_start=True)
            # Sanity checking if adversarial example is "legal"
            # assert(torch.max(torch.abs(white_adv_data-data)) <= (ATK_EPS + 1e-5) )
            assert(white_adv_data.max() == 1.)
            assert(white_adv_data.min() == 0.)
            if(batch_idx == 0):
                #remember to change the shuffle to false
                att.plot_attention(data.detach().cpu(),device,labels.cpu().detach().numpy(),whitebox,5,"./fig/beforewhitebox")
                att.plot_attention(white_adv_data.detach().cpu(),device,labels.cpu().detach().numpy(),whitebox,5,"./fig/whitebox")
            # Compute accuracy on perturbed data
            with torch.no_grad():
                # Stat keeping - whitebox
                whitebox_outputs = whitebox(white_adv_data)
                _,whitebox_preds = whitebox_outputs.max(1)
                whitebox_correct += whitebox_preds.eq(labels).sum().item()
                running_total += labels.size(0)

        # Print final 
        whitebox_acc = whitebox_correct/running_total
        white_list.append(whitebox_acc)
        print("PGD Attack Epsilon: {}; Whitebox Accuracy: {};".format(ATK_EPS, whitebox_acc))
        
    # plt.close()
    plt.plot(eps_list,white_list,label = "PGD")
    plt.title("whitebox")
    plt.xlabel("epsilon")
    plt.ylabel("accuracy")
    plt.ylim([0,1])
    plt.legend()
    plt.savefig("4.png")
    print("Done!")
def transfer_blackbox(device,val_loader,K):
    netA = hw2.ResNets()
    netB = ResNets(K)
    netB.load_state_dict(torch.load('./saved_model/ResNets.pth')['state_dict'])
    netA.load_state_dict(torch.load('./saved_model/original.pth')['state_dict'])
    netB = netB.to(device)
    netA = netA.to(device)
    eps_list = np.linspace(0,0.1,11)
    netA.eval()
    netB.eval()
    netB_list = []
    for ATK_EPS in eps_list:
        netB_correct = 0.
        running_total = 0.
        for batch_idx,(data,labels) in enumerate(val_loader):
            data = data.to(device) 
            labels = labels.to(device)
            # adv_data = attacks.random_noise_attack(whitebox,device,data.clone().detach(),ATK_EPS)
            ITS=10
            ALP = 1.85*(ATK_EPS/ITS) 
            white_adv_data = attacks.PGD_attack(netA,device,data.clone().detach(),labels,eps=ATK_EPS,alpha=ALP,iters=ITS,rand_start=True)
            # if(batch_idx == 0):
            #     att.plot_attention(white_adv_data.detach().cpu(),device,labels.cpu().detach().numpy(),netB,5,"./fig/transfer")
            #     att.plot_attention(data.detach().cpu(),device,labels.cpu().detach().numpy(),netB,5,"./fig/beforetransfer")
            # exit()
            # Compute accuracy on perturbed data
            with torch.no_grad():
                # Stat keeping - whitebox
                netB_outputs = netB(white_adv_data)
                _,netB_preds = netB_outputs.max(1)
                netB_correct += netB_preds.eq(labels).sum().item()
                running_total += labels.size(0)

        # Print final 
        netB_acc = netB_correct/running_total
        netB_list.append(netB_acc)
        print("PGD Attack Epsilon: {}; Whitebox Accuracy: {};".format(ATK_EPS, netB_acc))
        
    # plt.close()
    plt.plot(eps_list,netB_list,label = "PGD")
    plt.title("whitebox")
    plt.xlabel("epsilon")
    plt.ylabel("accuracy")
    plt.ylim([0,1])
    plt.legend()
    plt.savefig("4.png")
    print("Done!")

def training(net,train_loader):
    model_checkpoint = "pretrained/adversarial.pt"
    # ## Basic training params
    INITIAL_LR = 0.1
    MOMENTUM = 0.9
    REG = 1e-4
    num_epochs = 50
    best_val_acc = 0
    EPS = 0.06
    optimizer = optim.SGD(net.parameters(), lr=INITIAL_LR, momentum=MOMENTUM,weight_decay=REG,nesterov=True)
    scheduler = WarmUP(optimizer,INITIAL_LR,30)
    # optimizer = torch.optim.Adam(net.parameters(), lr=initial_lr)
    print("start adversarial training")
    for epoch in range(num_epochs):
        net.train()
        current_learning_rate = optimizer.param_groups[0]['lr']
        for batch_idx,(data,labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)
            ITS=10
            ALP = 1.85*(EPS/ITS)
            
            adv_data = attacks.PGD_attack(net,device,data,labels,eps=EPS,alpha=ALP,iters=ITS,rand_start=True)
            # Forward pass
            # print("get data")
            outputs = net(adv_data)
            # print("after forward")
            loss = criterion(outputs,labels)
            optimizer.zero_grad()
            # print("backward")
            loss.backward()
            optimizer.step()
            # Compute loss, gradients, and update params
            print("batch idx",batch_idx)
        
        scheduler.step()
        net.update_temp(epoch+1)
        test_acc,test_loss = test_model(net,val_loader,device)
        print('testing acc: %.5f, epoch: %d'%(test_acc,epoch))
        if test_acc > best_val_acc:
            best_val_acc = test_acc
            state = {'state_dict': net.state_dict(),
                 'epoch': epoch,
                 'lr': current_learning_rate}
            torch.save(state, model_checkpoint)
if __name__ == "__main__":

    DATA_ROOT = "./data"
    TRAIN_BATCH_SIZE = 256
    VAL_BATCH_SIZE = 100
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
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    # Model Definition  
    kernel_num = 5
    print("running with kernel size %d"%(kernel_num))
    print('Using device:', device)
    INITIAL_LR = 0.1
    EPOCHS = 100
    MOMENTUM = 0.9
    REG = 1e-4
    criterion = nn.CrossEntropyLoss()


    net = ResNets(kernel_num)
    net = net.to(device)
    if(sys.argv[1] == "1"):
        predict(net,val_loader,device)
    elif(sys.argv[1] == "2"):
        training(net,train_loader)
    elif(sys.argv[1] =="3"):
        transfer_blackbox(device,val_loader,kernel_num)

