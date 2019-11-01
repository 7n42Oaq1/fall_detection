import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision import models
import math

class Multiview_1(nn.Module):
    def __init__(self):
        super(Multiview_1, self).__init__()
        self.cnn1 = BasicCNN()
        self.cnn2 = BasicCNN()
        self.classifier = nn.Sequential(
            nn.Linear(2*3*3*64, 1024),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(1024, 13)
        )

    def forward(self, x):
        acc_x = x[:,0,:,:,:].squeeze(1)
        gyro_x = x[:,1,:,:,:].squeeze(1)
        acc_out = self.cnn1(acc_x)
        acc_out = acc_out.view(acc_out.size(0), -1)
        gyro_out = self.cnn2(gyro_x)
        gyro_out = gyro_out.view(gyro_out.size(0), -1)
        feature = torch.cat((acc_out, gyro_out), 1)
        out = self.classifier(feature)
        return out

class Multiview_2(nn.Module):
    def __init__(self):
        super(Multiview_2, self).__init__()
        self.cnn1 = BasicCNN()
        self.cnn2 = BasicCNN()
        #####
        self.classifier = nn.Sequential(
            nn.Linear(3*3*64, 1024),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(1024, 13)
        )

    def forward(self, x):
        acc_x = x[:,0,:,:,:].squeeze(1)
        gyro_x = x[:,1,:,:,:].squeeze(1)
        acc_out = self.cnn1(acc_x)
        gyro_out = self.cnn2(gyro_x)
        
        #  change from here
        # acc_out = acc_out.view(acc_out.size(0), -1)
        # gyro_out = gyro_out.view(gyro_out.size(0), -1)
        feature = torch.max(acc_out, gyro_out)
        feature = nn.MaxPool2d(kernel_size=2, stride=2)(feature)
        feature = feature.view(feature.size(0), -1)
        out = self.classifier(feature)
        return out

class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        # 20 * 20
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)            
        )
        # 10 * 10
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(0.25)            
        ) 
        # 8 * 8
        # self.layer3 = nn.Sequential(
        #     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
        #     # nn.InstanceNorm2d(64),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Dropout(0.25)            
        # ) 
        # 3 * 3
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        # out = self.layer3(out)
        return out

def weight_init(m):
# 使用isinstance来判断m属于什么类型
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
# m中的weight，bias其实都是Variable，为了能学习参数以及后向传播
        m.weight.data.fill_(1)
        m.bias.data.zero_()

if __name__ == "__main__":
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    # net = BasicCNN()
    # net = net.to(device)
    # summary(net, (3, 20, 20))
    net = Multiview_1()
    net = net.to(device)
    summary(net, (2, 3, 20, 20))