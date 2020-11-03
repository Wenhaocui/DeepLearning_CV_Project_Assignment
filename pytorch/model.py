import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

model_resnet50 = resnet50(pretrained=True)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding = 1)       
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding = 1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding = 1)        
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding = 1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding = 1)        
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding = 1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding = 1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding = 1)        
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding = 1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding = 1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding = 1)        
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding = 1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding = 1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout2d(0.5)
        
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 153)
        
    def forward(self, x):
        out = x
        
        # # handle single images
        # if (len(out.shape) == 3):
        #     out = out.unsqueeze(0)
        
        out = F.relu(self.conv1_1(out))
        out = F.relu(self.conv1_2(out))
        out = self.pool1(out)

        out = F.relu(self.conv2_1(out))
        out = F.relu(self.conv2_2(out))
        out = self.pool2(out)

        out = F.relu(self.conv3_1(out))
        out = F.relu(self.conv3_2(out))
        out = F.relu(self.conv3_3(out))
        out = self.pool3(out)

        out = F.relu(self.conv4_1(out))
        out = F.relu(self.conv4_2(out))
        out = F.relu(self.conv4_3(out))
        out = self.pool4(out)

        out = F.relu(self.conv5_1(out))
        out = F.relu(self.conv5_2(out))
        out = F.relu(self.conv5_3(out))
        out = self.pool5(out)
        
        # flatten
        out = out.view(-1, 512 * 7 * 7)
               
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        out = F.log_softmax(out, dim=1)
        
        return out

