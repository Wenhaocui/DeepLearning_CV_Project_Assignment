import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

model_resnet50 = resnet50(pretrained=True)

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
        linear_input = torch.flatten(feature, 1)
        #linear_input = feature.view(feature.size(0), -1)
        out_put = self.classifier(linear_input)
        return out_put
