import numpy as np
import os
import os.path as osp
import argparse

Config ={}
#Config['root_path'] = '/Users/chris/Desktop/EE599DEEP LEARNING/HW4/polyvore_outfits'
Config['root_path'] = '/mnt/polyvore_outfits'
Config['meta_file'] = 'polyvore_item_metadata.json'
Config['test_txt'] = 'test_category_hw.txt'
Config['out_file'] = 'test_category_output.txt'
Config['out_file2'] = 'test_category_output2.txt'
Config['checkpoint_path'] = ''


Config['use_cuda'] = True
Config['debug'] = False
Config['num_epochs'] = 20
Config['batch_size'] = 64

Config['learning_rate'] = 0.001
Config['num_workers'] = 5

Config['pretrained'] = False
