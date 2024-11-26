import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS
from utils.logger import *

@DATASETS.register_module()
class BasicShape(data.Dataset):
    def __init__(self, config):
        self.data_root = '/root/autodl-tmp/DataSet/BasicShape6/' # config.DATA_PATH
        self.pc_path = '/root/autodl-tmp/DataSet/BasicShape6/'
        self.subset = config.subset
        self.npoints = config.N_POINTS
        
        self.data_list_file = os.path.join(self.data_root,'BasicShape_' + f'{self.subset}.txt')
        test_data_list_file = os.path.join(self.data_root, 'BasicShape_test.txt')
        
        self.sample_points_num = config.npoints
        self.whole = config.get('whole')

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger = 'BasicShape')
        print_log(f'[DATASET] Open file {self.data_list_file}', logger = 'BasicShape')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        if self.whole:
            with open(test_data_list_file, 'r') as f:
                test_lines = f.readlines()
            print_log(f'[DATASET] Open file {test_data_list_file}', logger = 'BasicShape')
            lines = test_lines + lines
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('_')[0]
            model_id = line.split('_')[1].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger = 'BasicShape')

        self.permutation = np.arange(self.npoints)
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
        

    def random_sample(self, pc, num):
        pernum = np.arange(pc.shape[0])
        np.random.shuffle(pernum)
        pc = pc[pernum[:num]]
        return pc[:, :3]
        
    def __getitem__(self, idx):
        sample = self.file_list[idx]

        data = np.loadtxt(os.path.join(self.pc_path, sample['taxonomy_id'], sample['file_path'] + '.txt'), delimiter=',').astype(np.float32)

        data = self.random_sample(data, self.sample_points_num)
        data = self.pc_norm(data)
        data = torch.from_numpy(data).float()
        return sample['taxonomy_id'], sample['model_id'], data

    def __len__(self):
        return len(self.file_list)