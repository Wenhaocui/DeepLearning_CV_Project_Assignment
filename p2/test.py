import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from model import model_mobilenet, MyModel
from data import polyvore_test_final, polyvore_dataset
import os
import os.path as osp
import json
from tqdm import tqdm

from utils import Config

device = torch.device('cuda:0' if torch.cuda.is_available() and Config['use_cuda'] else 'cpu')

with open(osp.join(Config['root_path'], Config['out_file']), 'w') as test_output:

    model = model_mobilenet
    fc_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(fc_features, 153)

    model.load_state_dict(torch.load(osp.join(Config['root_path'], 'mobilenet.pth')))
    # model = MyModel()
    # model.load_state_dict(torch.load(osp.join(Config['root_path'], 'vgg16.pth')))

    model = model.to(device)
    model.eval()

    X_test, y_test = polyvore_dataset().create_testset()
    test_dataset = polyvore_test_final(X_test, y_test, polyvore_dataset().get_data_transforms()['test'])
    dataloader = DataLoader(test_dataset, shuffle=False, num_workers=Config['num_workers'])
    for input_name, inputs, labels in tqdm(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, pred = torch.max(outputs, 1)

        line = ""
        line = str(input_name[0][:-4]) +' '+ str(pred[0].item()) +' '+ str(labels[0].item()) + '\n'
        test_output.write(line)
