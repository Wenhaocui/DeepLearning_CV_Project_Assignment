import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, resnet50, vgg11

model_mobilenet = mobilenet_v2(pretrained=True)

cfgs =  [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def make_layers():
    layers = []
    in_channel = 3
    for cfg in cfgs:
        if cfg == 'M':
            layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
        else:
            conv2d = nn.Conv2d(in_channels=in_channel, out_channels=cfg, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(cfg), nn.ReLU(inplace=True)]
            in_channel = cfg
    return nn.Sequential(*layers)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.features = make_layers()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512 * 7 *7, out_features = 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 153)
        )

    def forward(self, input):
        feature = self.features(input)
        # linear_input = torch.flatten(feature, 1)
        linear_input = feature.view(feature.size(0), -1)
        out_put = self.classifier(linear_input)
        return out_put


# #pairwise model 
# def get_img_output_length(width, height):
#     def get_output_length(input_length):
#         # input_length += 6
#         filter_sizes = [2, 2, 2, 2, 2]
#         padding = [0, 0, 0, 0, 0]
#         stride = 2
#         for i in range(5):
#             input_length = (input_length+2*padding[i]-filter_sizes[i]) // stride + 1
#         return input_length
#     return get_output_length(width)*get_output_length(height) 
    
# class compModel(nn.Module):
#     def __init__(self, input_shape, pretrained=False):
#         super(compModel, self).__init__()
#         self.vgg = vgg11(pretrained=True)
#         del self.vgg.avgpool
#         del self.vgg.classifier
        
#         flat_shape = 512 * get_img_output_length(input_shape[1],input_shape[0])
#         self.fully_connect1 = torch.nn.Linear(flat_shape,512)
#         self.fully_connect2 = torch.nn.Linear(512,1)

#     def forward(self, x):
#         x1, x2 = x
#         x1 = self.vgg.features(x1)
#         x2 = self.vgg.features(x2)
#         b, _, _, _ = x1.size()        
#         x1 = x1.view([b,-1])
#         x2 = x2.view([b,-1])
#         x = torch.abs(x1-x2)
#         x = self.fully_connect1(x)
#         x = self.fully_connect2(x)
#         return x