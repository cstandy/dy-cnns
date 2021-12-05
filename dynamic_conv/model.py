import os, sys
import time
import datetime
from tqdm import tqdm_notebook as tqdm
from torch.optim.lr_scheduler import _LRScheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class AttentionLayer(nn.Module):
    def __init__(self, input, kernel,temperature=30):
        super().__init__()
        hidden = max(1, input // 4)
        # print("attention layer ",input)
        self.ave_pooling = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fc1 = nn.Linear(input, hidden)
        self.fc2 = nn.Linear(hidden, kernel)
        self.temp = temperature
    def forward(self, x):
        out = self.ave_pooling(x)
        out = self.fc1(out)
        out = self.fc2(F.relu(out))
        # print("first ",out.shape)
        out = F.softmax(out / self.temp, dim=-1)
        # print("after ",out.shape)
        return out
    def update_temp(self,epoch):
        if epoch < 10:
            self.temp -= 3
            # print('temperature =', self.temperature)
        else:
            self.temp = 1
class DynamicConv(nn.Module):
    def __init__(self,in_channel,out_channel,num_kernel,kernel_size,stride,padding,temp = 30):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.padding = padding
        self.num_kernel = num_kernel
        self.temp = temp
        self.kernel = kernel_size
        self.weight = nn.Parameter(torch.Tensor(num_kernel,out_channel,in_channel,kernel_size,kernel_size),requires_grad=True)
        self.bias = nn.Parameter(torch.Tensor(num_kernel,out_channel),requires_grad=True)
        self.attention = AttentionLayer(in_channel,num_kernel)
        # for initializatin
        for i_kernel in range(self.num_kernel):
            nn.init.kaiming_uniform_(self.weight[i_kernel], a=math.sqrt(5))
        bound = 1 / math.sqrt(self.weight[0, 0].numel())
        nn.init.uniform_(self.bias, -bound, bound)
    def update_temp(self,epoch):
        self.attention.update_temp(epoch)
    def forward(self,x):
        batch_size = x.shape[0]
        pi = self.attention(x)
        # print(x.shape)
        self.attention_score = pi
        # print(pi.shape)
        # input()
        weights = torch.sum(torch.mul(self.weight.unsqueeze(0),pi.view(batch_size,-1,1,1,1,1)),dim=1).view(-1,self.in_channel,self.kernel,self.kernel) 
        bias = torch.sum(torch.mul(self.bias.unsqueeze(0),pi.view(batch_size,-1,1)),dim=1).view(-1)
        x_grouped = x.view(1, -1, *x.shape[-2:]) 
        out = F.conv2d(x_grouped,weight = weights, bias = bias,stride = self.stride,padding = self.padding,groups=batch_size)
        return out.view(batch_size,self.out_channel,out.size(-2),out.size(-1))
class BasicBlock(nn.Module):
    expansion = 4
    def __init__(self,input1,output,kernel,s,num_kernel,downsample=None):
        super(BasicBlock,self).__init__()
        # self.conv1 = nn.Conv2d(input1,output,kernel,stride =s ,padding = 1)
        self.conv1 = DynamicConv(in_channel=input1,out_channel=output,num_kernel=num_kernel,kernel_size=kernel,stride =s ,padding = 1)
        self.conv1bn = nn.BatchNorm2d(output)
        # self.conv2 = nn.Conv2d(output,output,kernel,stride =1 ,padding = 1)
        self.conv2 = DynamicConv(in_channel=output,out_channel=output,num_kernel=num_kernel,kernel_size=kernel,stride =1 ,padding = 1)
        self.conv2bn = nn.BatchNorm2d(output)
        self.shortcut = nn.Sequential()
        self.out_filter = output
        self.stride = s
        self.downsample = downsample
    def update_temp(self,epoch):
        self.conv1.update_temp(epoch)
        self.conv2.update_temp(epoch)
    def forward(self,x):
        out = self.conv1(x)
        out = F.relu(self.conv1bn(out))
        out = self.conv2(out)
        residual = x
        if self.downsample:
            residual = self.downsample(x)
    
        out = self.conv2bn(out) + residual
        out = F.relu(out)
        return out
class ResNets(nn.Module):
    def __init__(self,num_kernel):
        super(ResNets,self).__init__()
        self.conv1 = nn.Conv2d(3,16,3,stride =1 ,padding = 1)
        self.block1 = self.make_layers(3,16,16,3,num_kernel,1)
        self.block2 = self.make_layers(3,16,32,3,num_kernel,2)
        self.block3 = self.make_layers(3,32,64,3,num_kernel,2)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(64,10)
    def make_layers(self, num_block, input1, output,kernel, num_kernel,s):
        layers = []
        downsample = None
        if s != 1 or input1 != output:
            downsample = nn.Sequential(DynamicConv(in_channel=input1,out_channel=output,kernel_size=kernel,num_kernel=num_kernel,stride =s ,padding = 1),nn.BatchNorm2d(output))
            # downsample = nn.Sequential(nn.Conv2d(input1, output, kernel, stride=s,padding=1),
            #     nn.BatchNorm2d(output))
        for i in range(num_block):
            if i == 0:
                layers.append(BasicBlock(input1=input1, output=output,kernel=kernel, s = s,num_kernel= num_kernel,downsample = downsample))
            else:
                layers.append(BasicBlock(input1=output, output=output,kernel=kernel,num_kernel=num_kernel, s = 1))
        return nn.Sequential(*layers)
    def update_temp(self,epoch):
        for module in self.modules():
            if isinstance(module,DynamicConv):
                module.update_temp(epoch)
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = F.avg_pool2d(out,out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)

        return out