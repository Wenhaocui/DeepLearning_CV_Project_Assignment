import os
from itertools import combinations
from utils import Config
import json

root_dir = Config['root_path']

def create_comp(json_file, input_file, comp_file):
    meta_file = open(os.path.join(root_dir, json_file),'r')
    meta_json = json.load(meta_file)
    set_to_items = {}
    for element in meta_json:
        set_to_items[element['set_id']] = element['items']

    file_write = open(os.path.join(root_dir, input_file),'w')

    with open(os.path.join(root_dir, comp_file),'r') as file_read:
        line = file_read.readline()
        while line:
            outfit = line.split()
            comb = list(combinations(list(range(1, len(outfit))), 2))
            for pair in comb:
                set1, idx1 = outfit[pair[0]].split('_')
                set2, idx2 = outfit[pair[1]].split('_')
                file_write.write(outfit[0] + ' ' + set_to_items[set1][int(idx1)-1]['item_id'] + ' ' +set_to_items[set2][int(idx2)-1]['item_id'] +'\n')

            line = file_read.readline()

if __name__=='__main__':
    create_comp('train.json', 'pairwise_compatibility_train.txt', 'compatibility_train.txt')
    create_comp('valid.json', 'pairwise_compatibility_valid.txt', 'compatibility_valid.txt')