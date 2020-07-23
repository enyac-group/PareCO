'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class MaskConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(MaskConv2d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)

        self.mask = nn.Parameter(torch.ones(out_channels), requires_grad=False)

    def forward(self, input):
        y = nn.functional.conv2d(
            input, self.weight, self.bias, self.stride, self.padding,
            self.dilation, self.groups)
        y = (y.permute(0,2,3,1) * self.mask).permute(0,3,1,2)
        return y

class DownsampleA(nn.Module):  
  def __init__(self, nIn, nOut, stride):
    super(DownsampleA, self).__init__() 
    assert stride == 2    
    self.out_channels = nOut
    self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)   

  def forward(self, x):   
    x = self.avg(x)  
    if self.out_channels-x.size(1) > 0:
        return torch.cat((x, torch.zeros(x.size(0), self.out_channels-x.size(1), x.size(2), x.size(3), device='cuda')), 1) 
    else:
        return x

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        MaskConv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv = nn.Sequential(
            MaskConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            MaskConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = DownsampleA(in_planes, planes, stride)
            #self.shortcut = nn.Sequential(
            #    nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
            #    nn.BatchNorm2d(planes)
            #)

    def forward(self, x):
        x = F.relu(self.shortcut(x) + self.conv(x))
        return x


class IRBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(IRBlock, self).__init__()
        expand_ratio = 6
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(in_planes, in_planes * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(in_planes * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(in_planes * expand_ratio, in_planes * expand_ratio, 3, stride, 1, groups=in_planes * expand_ratio, bias=False),
            nn.BatchNorm2d(in_planes * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(in_planes * expand_ratio, planes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(planes),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = DownsampleA(in_planes, planes, stride)
            #self.shortcut = nn.Sequential(
            #    nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
            #    nn.BatchNorm2d(planes)
            #)

    def forward(self, x):
        x = self.shortcut(x) + self.conv(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, width_mult=1):
        super(ResNet, self).__init__()
        self.in_planes = int(16*width_mult)

        self.features = [conv_bn(3, int(16*width_mult), 1)]
        self.features.append(self._make_layer(block, int(16*width_mult), num_blocks[0], stride=1))
        self.features.append(self._make_layer(block, int(32*width_mult), num_blocks[1], stride=2))
        self.features.append(self._make_layer(block, int(64*width_mult), num_blocks[2], stride=2))
        self.features.append(nn.AvgPool2d(8))
        self.features = nn.Sequential(*self.features)
        
        self.classifier = nn.Sequential(nn.Linear(int(64*width_mult), num_classes))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if type(m) in [nn.Conv2d, nn.Linear, nn.BatchNorm2d]:
                m.reset_parameters()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.classifier[0].in_features)
        x = self.classifier(x)
        return x

def ResNet8(num_classes=10, width_mult=1):
    return ResNet(BasicBlock, [1,1,1], num_classes=num_classes, width_mult=width_mult)

def ResNet20(num_classes=10, width_mult=1):
    return ResNet(BasicBlock, [3,3,3], num_classes=num_classes, width_mult=width_mult)

def ResNet32(num_classes=10, width_mult=1):
    return ResNet(BasicBlock, [5,5,5], num_classes=num_classes, width_mult=width_mult)

def ResNet44(num_classes=10, width_mult=1):
    return ResNet(BasicBlock, [7,7,7], num_classes=num_classes, width_mult=width_mult)

def ResNet56(num_classes=10, groups=None, width_mult=1):
    return ResNet(BasicBlock, [9,9,9], num_classes=num_classes, width_mult=width_mult)

def ResNet110(num_classes=10, groups=None, width_mult=1):
    return ResNet(BasicBlock, [18,18,18], num_classes=num_classes, width_mult=width_mult)

def IRBResNet56(num_classes=10, groups=None, width_mult=1):
    return ResNet(IRBlock, [9,9,9], num_classes=num_classes, width_mult=width_mult)


def test():
    net = ResNet20()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
