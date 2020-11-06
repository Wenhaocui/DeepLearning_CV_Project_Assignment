import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from model import model_mobilenet, MyModel
from data import polyvore_comp_test, polyvore_dataset
import os
import os.path as osp
import json
from tqdm import tqdm

from utils import Config

device = torch.device('cuda:0' if torch.cuda.is_available() and Config['use_cuda'] else 'cpu')

with open(osp.join(Config['root_path'], Config['out_compatability']), 'w') as test_output:

    model = model_mobilenet
    model.features[0][0] = nn.Conv2d(6, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    fc_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(fc_features, 1),
        nn.Sigmoid())

    model.load_state_dict(torch.load(osp.join(Config['root_path'], 'pairwise.pth')))
    

    model = model.to(device)
    model.eval()

    X_test = polyvore_dataset().create_comp_test()
    test_dataset = polyvore_comp_test(X_test, polyvore_dataset().get_data_transforms()['test'])
    dataloader = DataLoader(test_dataset, shuffle=False, num_workers=Config['num_workers'])
    for input_name, inputs in tqdm(dataloader):
        inputs = inputs.to(device)

        outputs = model(inputs)
        if outputs[0][0].item() >= 0.5:
            pred = 1.0
        else:
            pred = 0.0
        line = ""
        line = str(input_name[0][0][:-4]) + ' ' + str(input_name[1][0][:-4]) + ' ' + str(outputs[0][0].item()) +' '+ str(pred) + '\n'
        test_output.write(line)
