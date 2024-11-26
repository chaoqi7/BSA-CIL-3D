import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
import os
from tqdm import tqdm
import torch
from utils.logger import *
import pickle
import h5py

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class ModelNet(iData):
    use_path = False

    def download_data(self, config = None):
        self.root = '/root/autodl-tmp/DataSet/modelnet40_normal_resampled/' #config.DATA_PATH
        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.uniform = True
        self.process_data = False
        self.npoints = 1024# config.N_POINTS
        self.use_normals = False #config.USE_NORMALS
        self.num_category = 40 #config.NUM_CATEGORY

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        shape_names_train = ['_'.join(x.split('_')[0:-1]) for x in shape_ids['train']]
        self.datapath_train = [(shape_names_train[i], os.path.join(self.root, shape_names_train[i], shape_ids['train'][i]) + '.txt') for i
                         in range(len(shape_ids['train']))]

        shape_names_test = ['_'.join(x.split('_')[0:-1]) for x in shape_ids['test']]
        self.datapath_test = [(shape_names_test[i], os.path.join(self.root, shape_names_test[i], shape_ids['test'][i]) + '.txt') for i
                         in range(len(shape_ids['test']))]

        print('The size of %s data is %d' % ('train', len(self.datapath_train)))
        print('The size of %s data is %d' % ('test', len(self.datapath_test)))

        if self.uniform:
            self.train_save_path = os.path.join(self.root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, 'train', self.npoints))
            self.test_save_path = os.path.join(self.root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, 'test', self.npoints))
        else:
            self.train_save_path = os.path.join(self.root,'modelnet%d_%s_%dpts.dat' % (self.num_category, 'train', self.npoints))
            self.test_save_path = os.path.join(self.root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, 'test', self.npoints))

        if (not os.path.exists(self.train_save_path)) or (not os.path.exists(self.test_save_path)):
            self.process_data = True

        if self.process_data:
            print('Processing data (only running in the first time)')
            # print_log('Processing data %s (only running in the first time)...' % self.save_path, logger='ModelNet')
            self.train_data = [None] * len(self.datapath_train)
            self.train_targets = [None] * len(self.datapath_train)
            self.test_data = [None] * len(self.datapath_test)
            self.test_targets = [None] * len(self.datapath_test)

            for index in tqdm(range(len(self.datapath_train)), total=len(self.datapath_train)):
                fn = self.datapath_train[index]
                cls = self.classes[self.datapath_train[index][0]]
                cls = np.array([cls]).astype(np.int32)
                point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                if self.uniform:
                    point_set = farthest_point_sample(point_set, self.npoints)
                else:
                    point_set = point_set[0:self.npoints, :]

                point_set[:, 0:3] = self.pc_normalize(point_set[:, 0:3])
                if not self.use_normals:
                    point_set = point_set[:, 0:3]
                self.train_data[index] = point_set
                self.train_targets[index] = cls

            with open(self.train_save_path, 'wb') as f:
                pickle.dump([self.train_data, self.train_targets], f)

            for index in tqdm(range(len(self.datapath_test)), total=len(self.datapath_test)):
                fn = self.datapath_test[index]
                cls = self.classes[self.datapath_test[index][0]]
                cls = np.array([cls]).astype(np.int32)
                point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                if self.uniform:
                    point_set = farthest_point_sample(point_set, self.npoints)
                else:
                    point_set = point_set[0:self.npoints, :]

                point_set[:, 0:3] = self.pc_normalize(point_set[:, 0:3])
                if not self.use_normals:
                    point_set = point_set[:, 0:3]
                self.test_data[index] = point_set
                self.test_targets[index] = cls

            with open(self.test_save_path, 'wb') as f:
                pickle.dump([self.test_data, self.test_targets], f)
        else:
            print('Load processed data')
            with open(self.train_save_path, 'rb') as f:
                self.train_data, self.train_targets = pickle.load(f)
            with open(self.test_save_path, 'rb') as f:
                self.test_data, self.test_targets = pickle.load(f)
        self.train_data = np.stack(self.train_data, axis=0)
        self.test_data = np.stack(self.test_data, axis=0)
        self.train_targets = np.stack(self.train_targets, axis=0).reshape(-1)
        self.test_targets = np.stack(self.test_targets, axis=0).reshape(-1)
        if not self.use_normals:
            self.train_data = self.train_data[:,:, 0:3]
            self.test_data = self.test_data[:, :, 0:3]
        #self.train_data = self.train_data.transpose(0, 2, 1)
        #self.test_data = self.test_data.transpose(0, 2, 1)
        print('Data Loading Finished..')

    def pc_normalize(self, pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

class ShapeNet(iData):
    use_path = False

    def download_data(self, config = None):
        self.root = '/root/autodl-tmp/DataSet/ShapeNet55/' #config.DATA_PATH
        self.uniform = True
        self.process_data = False
        self.npoints = 1024  # config.N_POINTS
        self.use_normals = False  # config.USE_NORMALS
        self.num_category = 55  # config.NUM_CATEGORY

        cLass_names = set()
        files = os.listdir(os.path.join(self.root, 'shapenet_pc'))
        for file in files:
            prefix = file.split('-')[0]
            cLass_names.add(prefix)
        self.cat = list(cLass_names)
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'test.txt'))]

        shape_names_train = [x for x in shape_ids['train']]
        self.datapath_train = [(shape_names_train[i], os.path.join(self.root, 'shapenet_pc' , shape_names_train[i])) for i
            in range(len(shape_ids['train']))]

        shape_names_test = [x for x in shape_ids['test']]
        self.datapath_test = [(shape_names_test[i], os.path.join(self.root, 'shapenet_pc' , shape_names_test[i])) for i
            in range(len(shape_ids['test']))]

        print('The size of %s data is %d' % ('train', len(shape_ids['train'])))
        print('The size of %s data is %d' % ('test', len(shape_ids['test'])))

        if self.uniform:
            self.train_save_path = os.path.join(self.root, 'shapenet%d_%s_%dpts_fps.dat' % (self.num_category, 'train', self.npoints))
            self.test_save_path = os.path.join(self.root, 'shapenet%d_%s_%dpts_fps.dat' % (self.num_category, 'test', self.npoints))
        else:
            self.train_save_path = os.path.join(self.root,'shapenet%d_%s_%dpts.dat' % (self.num_category, 'train', self.npoints))
            self.test_save_path = os.path.join(self.root, 'shapene%d_%s_%dpts.dat' % (self.num_category, 'test', self.npoints))

        if (not os.path.exists(self.train_save_path)) or (not os.path.exists(self.test_save_path)):
            self.process_data = True

        if self.process_data:
            print('Processing data (only running in the first time)')
            # print_log('Processing data %s (only running in the first time)...' % self.save_path, logger='ModelNet')
            self.train_data = [None] * len(self.datapath_train)
            self.train_targets = [None] * len(self.datapath_train)
            self.test_data = [None] * len(self.datapath_test)
            self.test_targets = [None] * len(self.datapath_test)

            for index in tqdm(range(len(self.datapath_train)), total=len(self.datapath_train)):
                fn = self.datapath_train[index]
                cls = self.classes[self.datapath_train[index][0].split('-')[0]]
                cls = np.array([cls]).astype(np.int32)
                point_set = np.load(fn[1]).astype(np.float32)

                if self.uniform:
                    point_set = farthest_point_sample(point_set, self.npoints)
                else:
                    point_set = point_set[0:self.npoints, :]

                point_set[:, 0:3] = self.pc_normalize(point_set[:, 0:3])
                if not self.use_normals:
                    point_set = point_set[:, 0:3]
                self.train_data[index] = point_set
                self.train_targets[index] = cls

            with open(self.train_save_path, 'wb') as f:
                pickle.dump([self.train_data, self.train_targets], f)

            for index in tqdm(range(len(self.datapath_test)), total=len(self.datapath_test)):
                fn = self.datapath_test[index]
                cls = self.classes[self.datapath_test[index][0].split('-')[0]]
                cls = np.array([cls]).astype(np.int32)
                point_set = np.load(fn[1]).astype(np.float32)

                if self.uniform:
                    point_set = farthest_point_sample(point_set, self.npoints)
                else:
                    point_set = point_set[0:self.npoints, :]

                point_set[:, 0:3] = self.pc_normalize(point_set[:, 0:3])
                if not self.use_normals:
                    point_set = point_set[:, 0:3]
                self.test_data[index] = point_set
                self.test_targets[index] = cls

            with open(self.test_save_path, 'wb') as f:
                pickle.dump([self.test_data, self.test_targets], f)
        else:
            print('Load processed data')
            with open(self.train_save_path, 'rb') as f:
                self.train_data, self.train_targets = pickle.load(f)
            with open(self.test_save_path, 'rb') as f:
                self.test_data, self.test_targets = pickle.load(f)
        self.train_data = np.stack(self.train_data, axis=0)
        self.test_data = np.stack(self.test_data, axis=0)
        self.train_targets = np.stack(self.train_targets, axis=0).reshape(-1)
        self.test_targets = np.stack(self.test_targets, axis=0).reshape(-1)
        if not self.use_normals:
            self.train_data = self.train_data[:,:, 0:3]
            self.test_data = self.test_data[:, :, 0:3]
        #self.train_data = self.train_data.transpose(0, 2, 1)
        #self.test_data = self.test_data.transpose(0, 2, 1)
        print('Data Loading Finished..')

    def pc_normalize(self, pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

class ScanObject(iData):
    use_path = False

    def download_data(self, config = None):
        self.root = '/root/autodl-tmp/DataSet/ScanObjectNN_h5_files/h5_files/main_split_nobg/'
        self.npoints = 1024  # config.N_POINTS
        self.uniform = True
        self.process_data = False
        self.num_category = 15

        self.train_load_path = os.path.join(self.root, 'training_objectdataset_augmentedrot.h5')
        self.test_load_path = os.path.join(self.root, 'test_objectdataset_augmentedrot.h5')

        if self.uniform:
            self.train_save_path = os.path.join(self.root, 'scanobject%d_%s_%dpts_fps.dat' % (self.num_category, 'train', self.npoints))
            self.test_save_path = os.path.join(self.root, 'scanobject%d_%s_%dpts_fps.dat' % (self.num_category, 'test', self.npoints))
        else:
            self.train_save_path = os.path.join(self.root,'scanobject%d_%s_%dpts.dat' % (self.num_category, 'train', self.npoints))
            self.test_save_path = os.path.join(self.root, 'scanobject%d_%s_%dpts.dat' % (self.num_category, 'test', self.npoints))

        if (not os.path.exists(self.train_save_path)) or (not os.path.exists(self.test_save_path)):
            self.process_data = True

        if self.process_data:
            print('Processing data (only running in the first time)')
            train_file = h5py.File(self.train_load_path, 'r')
            test_file = h5py.File(self.test_load_path, 'r')
            self.train_data = [None] * len(train_file['data'])
            self.train_targets = [None] * len(train_file['label'])
            self.test_data = [None] * len(test_file['data'])
            self.test_targets = [None] * len(test_file['label'])
            train_data_tmpt = np.array(train_file['data']).astype(np.float32)
            train_targets_tmpt = np.array(train_file['label']).astype(int)
            test_data_tmpt = np.array(test_file['data']).astype(np.float32)
            test_targets_tmpt = np.array(test_file['label']).astype(int)

            for index in tqdm(range(len(train_data_tmpt)), total=len(train_data_tmpt)):
                if self.uniform:
                    point_set = farthest_point_sample(train_data_tmpt[index], self.npoints)
                else:
                    point_set = train_data_tmpt[index][0:self.npoints, :]
                point_set[:, 0:3] = self.pc_normalize(point_set[:, 0:3])
                point_set = point_set[:, 0:3]
                self.train_data[index] = point_set
                self.train_targets[index] = train_targets_tmpt[index]

            with open(self.train_save_path, 'wb') as f:
                pickle.dump([self.train_data, self.train_targets], f)

            for index in tqdm(range(len(test_data_tmpt)), total=len(test_data_tmpt)):
                if self.uniform:
                    point_set = farthest_point_sample(test_data_tmpt[index], self.npoints)
                else:
                    point_set = test_data_tmpt[index][0:self.npoints, :]
                point_set[:, 0:3] = self.pc_normalize(point_set[:, 0:3])
                point_set = point_set[:, 0:3]
                self.test_data[index] = point_set
                self.test_targets[index] = test_targets_tmpt[index]

            with open(self.test_save_path, 'wb') as f:
                pickle.dump([self.test_data, self.test_targets], f)

        else:
            print('Load processed data')
            with open(self.train_save_path, 'rb') as f:
                self.train_data, self.train_targets = pickle.load(f)
            with open(self.test_save_path, 'rb') as f:
                self.test_data, self.test_targets = pickle.load(f)
        self.train_data = np.stack(self.train_data, axis=0)
        self.test_data = np.stack(self.test_data, axis=0)
        self.train_targets = np.stack(self.train_targets, axis=0).reshape(-1)
        self.test_targets = np.stack(self.test_targets, axis=0).reshape(-1)
        self.train_data = self.train_data[:,:, 0:3]
        self.test_data = self.test_data[:, :, 0:3]
        #self.train_data = self.train_data.transpose(0, 2, 1)
        #self.test_data = self.test_data.transpose(0, 2, 1)
        print('Data Loading Finished..')

    def pc_normalize(self, pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc
