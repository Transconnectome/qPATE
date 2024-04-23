import torch
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data_utils
from PIL import Image

import math

from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split, StratifiedKFold

from utils.utils import imshow

from typing import Dict, List
from collections.abc import Callable


    
def get_mnist_data(args):

    mnist_path = "/global/homes/h/heehaw/qPATE_GAN_lightning/data/MNIST/" # for loading and saving tensors  
    x_train = torch.load(mnist_path + 'x_train.pt')
    # image transforms
    transform = transforms.Compose([
    transforms.Resize(32),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,),(0.5,)),
    ])

    to_pil_image = transforms.ToPILImage()


    train_data = datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.MNIST(
        root='data',
        train=False,
        download=True,
        transform=transform
    )

    train_filter = np.where((train_data.targets == 0 ) | (train_data.targets == 1))
    test_filter = np.where((test_data.targets == 0) | (test_data.targets == 1))

    train_data.data, train_data.targets = train_data.data[train_filter], train_data.targets[train_filter]
    test_data.data, test_data.targets = test_data.data[test_filter], test_data.targets[test_filter]        

    #data = torch.utils.data.ConcatDataset([train_data, test_data])    
    size = {}
    size['t_train'] = int(args.n_teachers * args.n_samples)
    size['t_val'] = int(args.n_teachers * (args.n_samples / 10))
    size['t_discard'] = int(len(train_data.data) - size['t_train'] - size['t_val'])
    size['s_train'] = int(1 * args.n_samples)
    size['s_val'] = int(1 * (args.n_samples / 10))
    size['s_test'] = int(1 * (args.n_samples / 10))
    size['s_discard'] = int(len(test_data.data) - size['s_train'] - size['s_val'])
    print("total dataset size:", int(len(train_data.data) + len(test_data.data)))
    print("t_train size:", size['t_train'])
    print("t_val size:", size['t_val'])
    print("s_train size:", size['s_train'])
    print("s_val size:", size['s_val'])
    print("s_test size:", size['s_test'])
    print("t_discard size:", size['t_discard'])
    print("s_discard size:", size['s_discard'])

    # Teacher dataset
    t_train_X, t_val_X, t_train_y, t_val_y = train_test_split(train_data.data, train_data.targets,
                                                      train_size=size['t_train'], 
                                                      test_size=size['t_val'], 
                                                      random_state=args.seed, shuffle=True, stratify=train_data.targets)
    # Student dataset 
    s_train_val_X, s_test_X, s_train_val_y, s_test_y = train_test_split(test_data.data, test_data.targets,
                                                      train_size=size['s_train']+size['s_val'], 
                                                      test_size=size['s_test'], 
                                                      random_state=args.seed, shuffle=True, stratify=test_data.targets)
    s_train_X, s_val_X, s_train_y, s_val_y = train_test_split(s_train_val_X, s_train_val_y,
                                                      train_size=size['s_train'], 
                                                      test_size=size['s_val'], 
                                                      random_state=args.seed, shuffle=True, stratify=s_train_val_y)
    if args.val_test_together_student is False:
        t_train_dataset = PATE_Dataset(data=t_train_X, targets=t_train_y, n_networks=args.n_teachers, train_teacher=True, train_student=False)
        t_val_dataset = PATE_Dataset(data=t_val_X, targets=t_val_y, n_networks=args.n_teachers, train_teacher=True, train_student=False)
        s_train_dataset = PATE_Dataset(data=s_train_X, targets=s_train_y, n_networks=1, train_teacher=False, train_student=True)
        s_val_dataset = PATE_Dataset(data=s_val_X, targets=s_val_y, n_networks=1, train_teacher=False, train_student=True)
        s_test_dataset = PATE_Dataset(data=s_test_X, targets=s_test_y, n_networks=1, train_teacher=False, train_student=True)
        partition = {} 
        partition['t_train'] = t_train_dataset
        partition['t_val'] = t_val_dataset
        partition['s_train'] = s_train_dataset
        partition['s_val'] = s_val_dataset
        partition['s_test'] = s_test_dataset
    else: 
        t_train_dataset = PATE_Dataset(data=t_train_X, targets=t_train_y, n_networks=args.n_teachers, train_teacher=True, train_student=False)
        t_val_dataset = PATE_Dataset(data=t_val_X, targets=t_val_y, n_networks=args.n_teachers, train_teacher=True, train_student=False)
        s_train_dataset = PATE_Dataset(data=s_train_X, targets=s_train_y, n_networks=1, train_teacher=False, train_student=True)
        s_val_test_dataset = PATE_Dataset(data={'val':s_val_X, 'test':s_test_X} , targets={'val':s_val_y, 'test':s_test_y} ,n_networks=1, train_teacher=False, train_student=True, val_test_together_student=True)     
        partition = {} 
        partition['t_train'] = t_train_dataset
        partition['t_val'] = t_val_dataset
        partition['s_train'] = s_train_dataset
        partition['s_val_test'] = s_val_test_dataset
    return partition


class PATE_Dataset(Dataset): 
    def __init__(self, data, targets, n_networks, train_teacher=True, train_student=False, val_test_together_student=False): 
        super(Dataset, self).__init__()
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.n_networks = n_networks
        self.train_teacher = train_teacher
        self.train_student = train_student
        self.val_test_together_student = val_test_together_student
        if val_test_together_student is False:
            self.total_samples = len(data)
            dataset = torch.utils.data.TensorDataset(data, targets)
        else: 
            assert len(data['val']) == len(data['test'])
            self.total_samples = len(data['val'])
            dataset = {'val': torch.utils.data.TensorDataset(data['val'], targets['val']), 'test': torch.utils.data.TensorDataset(data['test'], targets['test'])}
        if self.train_teacher: 
            self.dataset = torch.utils.data.random_split(dataset, [int(len(dataset) / self.n_networks) for _ in range(self.n_networks)])
        elif self.n_networks == 1: 
            self.dataset = dataset 
        ##TODO 
        # stratified k splits
        self.data = data 
        self.targets = targets

    def __len__(self) -> int: 
        return int(self.total_samples / self.n_networks)

    def __getitem__(self, index: int) :
        '''
        batch = [{'t_train':
                      {'teacher1':[(img, label), ...], teacher2':[(img, label), ...], ..., teacherN':[(img, label), ...]}, 
                  's_train_l':[(img, noisy_label), ...], 
                  's_train_u': [(img, label), ...]}]
        '''
        if self.train_teacher: 
            data = {}
            # train data for teachers
            for ith in range(self.n_networks):
                img, label = self.dataset[ith][index]
                data['teacher%s'%ith] = (img.float().unsqueeze(0), label)
        elif self.train_student: 
            if self.val_test_together_student is False:
                img, label = self.dataset[index]
                data = (img.float().unsqueeze(0), label)
            else: 
                data = {}
                val_img, val_label = self.dataset['val'][index]
                test_img, test_label = self.dataset['test'][index]
                data['val'] = (val_img.float().unsqueeze(0), val_label)
                data['test'] = (test_img.float().unsqueeze(0), test_label)
        return data
    
