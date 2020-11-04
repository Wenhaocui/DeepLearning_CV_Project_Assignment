import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from model import model_mobilenet, MyModel
from data import polyvore_test, polyvore_dataset
import os
import numpy as np
import os.path as osp
import json
from tqdm import tqdm
from PIL import Image

from utils import Config

def create_testset():

    meta_file = open(osp.join(Config['root_path'], Config['meta_file']), 'r')
    meta_json = json.load(meta_file)
    id_to_category = {}
    for k, v in meta_json.items():
        id_to_category[k] = v['category_id'] 

    test_list = []
    test_file = open(osp.join(Config['root_path'], Config['test_txt']), 'r')
    lines = test_file.readlines() 
    for line in lines:
        line = line.strip()
        test_list.append(line)

    image_dir = osp.join(Config['root_path'], 'images')
    files = os.listdir(image_dir)
    files_set = set(map(lambda x: x[:-4], files))
    X = []; y = []
    for x in test_list:
        if x in files_set and x in id_to_category:
            X.append(x+".jpg")
            y.append(int(id_to_category[x]))
    y = LabelEncoder().fit_transform(y)

    return X, y

device = torch.device('cuda:0' if torch.cuda.is_available() and Config['use_cuda'] else 'cpu')

if Config['pretrained'] == True:
    model = model_mobilenet
    fc_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(fc_features, 153)

    model.load_state_dict(torch.load('/mnt/polyvore_outfits/mobilenet.pth'))
    model.to(device)
else:
    model = MyModel
    model.load_state_dict(torch.load('/mnt/polyvore_outfits/vgg16.pth'))
model.eval()

X_test, y_test = create_testset()
test_dataset = polyvore_test(X_test, y_test, polyvore_dataset().get_data_transforms()['test'])
dataloader = DataLoader(test_dataset, shuffle=False,batch_size=len(X_test))
test_output = open(osp.join(Config['root_path'], Config['out_file']), 'w')
for i, inputs, labels in tqdm(enumerate(dataloader)):
    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = model(inputs)
    _, pred = torch.max(outputs, 1)

    line = ""
    line = str(X_test[i]) +'  '+ str(pred) +'  '+ str(labels) + '\n'
    test_output.writeline(line)

test_output.close()
