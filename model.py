import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=1,bias=False)


class DQN(nn.Module):
    def __init__(self, action=12):
        super(DQN, self).__init__()

        self.conv_init = nn.Conv2d(3, 16, kernel_size=5,stride=1,padding=2,bias=False)
        self.pool0 = torch.nn.MaxPool2d(kernel_size = 4)
        self.conv1 = conv3x3(16,16)
        self.conv2 = conv3x3(16,16)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = torch.nn.MaxPool2d(kernel_size = 2)

        self.conv3 = conv3x3(16,32)
        self.conv4 = conv3x3(32,32)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool2 = torch.nn.MaxPool2d(kernel_size = 2)

        self.conv5 = conv3x3(32,64)
        self.conv6 = conv3x3(64,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool3 = torch.nn.MaxPool2d(kernel_size = 2)

        self.conv7 = conv3x3(64,128)
        self.conv8 = conv3x3(128,128)
        self.bn1 = nn.BatchNorm2d(128)
        self.pool4 = torch.nn.MaxPool2d(kernel_size = 2)
        self.conv9 = conv3x3(128,256)
        self.conv10 = conv3x3(256,256)
        self.bn1 = nn.BatchNorm2d(256)
        self.head = nn.Linear(256, action)
        
    def forward(self, screen):
        x = self.pool0(screen)

        x = F.relu(self.conv_init(x))

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)


        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)


        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)


        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool4(x)

        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))

        x = F.avg_pool2d(x, kernel_size = x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x






