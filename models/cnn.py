import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os


class Conv(nn.Sequential):
    def __init__(self, inch, outch, kernel=3, stride=1, padding=1, bias=False, act='relu', *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_module('Conv', nn.Conv2d(in_channels=inch, out_channels=outch, kernel_size=kernel, padding=padding, bias=bias, stride=stride))
        self.add_module('BatchNorm', nn.BatchNorm2d(outch))
        if act == "relu":
            self.add_module("relu", module=nn.ReLU())
        elif act == "leaky":
            self.add_module("leaky", module=nn.LeakyReLU())
        elif act == "relu6":
            self.add_module("relu6", module=nn.ReLU6(True))
        elif act == "tanh":
            self.add_module("tanh", module=nn.Tanh())
        else:
            raise("Activation function not implemented")
        
    def forward(self, x):
        return super().forward(x)
    
class GlobalAveragePool(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self, x):
        _, _, h, _ = x.shape
        return F.avg_pool2d(x, kernel_size=h)
    
class CNNBlock(nn.Module):
    def __init__(self, inch, midch, outch, kernel=3, stride=1, padding=1, bias=False, act='relu', dropout=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if dropout:
            self.layers1 = nn.Sequential(
                Conv(inch=inch, outch=outch, kernel=kernel, stride=stride, padding=padding, bias=bias, act=act),
                nn.Dropout(dropout),
            )

            self.layers2 = nn.Sequential(
                Conv(inch=inch, outch=midch, kernel=kernel, stride=stride, padding=padding, bias=bias, act=act),
                Conv(inch=midch, outch=outch, kernel=kernel, stride=stride, padding=padding, bias=bias, act=act),
                nn.MaxPool2d(kernel_size=kernel),
                nn.Dropout(dropout),
            )
        else:
            self.layers1 = Conv(inch=inch, outch=outch, kernel=kernel, stride=stride, padding=padding, bias=bias, act=act)

            self.layers2 = nn.Sequential(
                Conv(inch=inch, outch=midch, kernel=kernel, stride=stride, padding=padding, bias=bias, act=act),
                Conv(inch=midch, outch=outch, kernel=kernel, stride=stride, padding=padding, bias=bias, act=act),
                nn.MaxPool2d(kernel_size=kernel),
            )                       
    def forward(self, x):
        x1 = self.layers1(x)
        x2 = self.layers2(x)
        out = x1 + x2
        return out
    
class CNN(nn.Module):
    def __init__(self, inch, midch, outch, softmax=False, n_blocks=4, kernel=3, stride=1, padding=1, dropout=0.1, bias=False, act='relu', *args, **kwargs):
        super().__init__(*args, **kwargs)

        ch_list = [16, 32, 64, 128, 256]
        kernel_list = [7, 5, 3, 3]
        i = 0
        ch = ch_list[i]
        self.softmax = softmax
        
        self.start = nn.Sequential(
            Conv(inch=inch, outch=ch, kernel=kernel, stride=stride, padding=padding, bias=bias, act=act),
            Conv(inch=ch, outch=ch, kernel=kernel, stride=stride, padding=padding, bias=bias, act=act)
        )
        i += 1
        self.blocks = nn.ModuleList([])
        for _ in range(n_blocks):
            ch = ch[i]
            self.blocks.append(nn.Conv2d(in_channels=ch_list[i-1], out_channels=ch, kernel_size=1, stride=1))
            self.blocks.append(CNNBlock(inch=ch, midch=ch, outch=ch, kernel=kernel, stride=stride, padding=padding, bias=bias, act=act, dropout=dropout))
            

        self.end = nn.Sequential(
            nn.Conv2d(in_channels=inch, out_channels=outch, kernel_size=1, stride=1),
            GlobalAveragePool()
        )

    def forward(self, x):
        x = self.start(x)

        for i in range(len(self.blocks)):
            layer = self.blocks[i]
            x = layer(x)

        x = self.end(x)
            
        if self.softmax:
            x = nn.Softmax(x)
        return x
    
if __name__ == '__main__':
    x = torch.rand(size=(1, 3, 192, 192))
    model = CNN()