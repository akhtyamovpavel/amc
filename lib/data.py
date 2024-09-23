# Code for "AMC: AutoML for Model Compression and Acceleration on Mobile Devices"
# Yihui He*, Ji Lin*, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han
# {jilin, songhan}@mit.edu

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

import os
from torch.utils.data import Dataset, DataLoader
from torchvision.models import MobileNet_V2_Weights
from sklearn.model_selection import train_test_split
from PIL import Image

class FlowersDatasetImageNet(Dataset):
    def __init__(self, paths, answers):
        super().__init__()
        self.paths = paths
        self.answers = answers
        self.transform = MobileNet_V2_Weights.IMAGENET1K_V1.transforms(crop_size=227)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        answer = self.answers[index]

        # image = Image.open(path)
        with Image.open(path) as image:
            input_tensor = self.transform(image)

        # image.close()

        return input_tensor, answer

def get_flower_paths():
    import os
    DATASET_FOLDER = "../flower_photos"
    class_names = sorted([
        class_name for class_name in os.listdir(DATASET_FOLDER) if not class_name.endswith('txt')
    ])

    CLASS_INDICES = {
        class_name: index for index, class_name in enumerate(class_names)
    }
    paths = []
    indices = []

    for class_name in CLASS_INDICES.keys():
        class_folder = os.path.join(DATASET_FOLDER, class_name)

        for filename in sorted(os.listdir(class_folder)):
            if not filename.endswith('jpg'):
                continue
            path = os.path.join(class_folder, filename)
            indices.append(CLASS_INDICES[class_name])
            paths.append(path)

    return paths, indices

def get_dataset(dset_name, batch_size, n_worker, data_root='../../data'):
    flower_paths, flower_indices = get_flower_paths()
    flower_paths_train, flower_paths_val, flower_y_train, flower_y_val = train_test_split(
        flower_paths,
        flower_indices,
        random_state=42,
        test_size=0.2
    )
    train_dataset_image_net = FlowersDatasetImageNet(paths=flower_paths_train, answers=flower_y_train)
    val_dataset_image_net = FlowersDatasetImageNet(paths=flower_paths_val, answers=flower_y_val)

    train_dataloader_image_net = DataLoader(
        train_dataset_image_net, batch_size=batch_size, shuffle=True, num_workers=n_worker
    )
    val_dataloader_image_net = DataLoader(
        val_dataset_image_net,
        batch_size=batch_size,
        shuffle=False, num_workers=n_worker
    )

    # cifar_tran_train = [
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ]
    # cifar_tran_test = [
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ]
    # print('=> Preparing data..')
    # if dset_name == 'cifar10':
    #     transform_train = transforms.Compose(cifar_tran_train)
    #     transform_test = transforms.Compose(cifar_tran_test)
    #     trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
    #     train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
    #                                                num_workers=n_worker, pin_memory=True, sampler=None)
    #     testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
    #     val_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
    #                                              num_workers=n_worker, pin_memory=True)
    #     n_class = 10
    # elif dset_name == 'imagenet':
    #     # get dir
    #     traindir = os.path.join(data_root, 'train')
    #     valdir = os.path.join(data_root, 'val')

    #     # preprocessing
    #     input_size = 224
    #     imagenet_tran_train = [
    #         transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0)),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ]
    #     imagenet_tran_test = [
    #         transforms.Resize(int(input_size / 0.875)),
    #         transforms.CenterCrop(input_size),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ]

    #     train_loader = torch.utils.data.DataLoader(
    #         datasets.ImageFolder(traindir, transforms.Compose(imagenet_tran_train)),
    #         batch_size=batch_size, shuffle=True,
    #         num_workers=n_worker, pin_memory=True, sampler=None)

    #     val_loader = torch.utils.data.DataLoader(
    #         datasets.ImageFolder(valdir, transforms.Compose(imagenet_tran_test)),
    #         batch_size=batch_size, shuffle=False,
    #         num_workers=n_worker, pin_memory=True)
    #     n_class = 1000

    # else:
    #     raise NotImplementedError

    return train_dataloader_image_net, val_dataloader_image_net, 5


def get_split_dataset(dset_name, batch_size, n_worker, val_size, data_root='../data',
                      use_real_val=False, shuffle=True):
    flower_paths, flower_indices = get_flower_paths()
    flower_paths_train, flower_paths_val, flower_y_train, flower_y_val = train_test_split(
        flower_paths,
        flower_indices,
        random_state=42,
        test_size=0.2
    )
    train_dataset_image_net = FlowersDatasetImageNet(paths=flower_paths_train, answers=flower_y_train)
    val_dataset_image_net = FlowersDatasetImageNet(paths=flower_paths_val, answers=flower_y_val)

    train_dataloader_image_net = DataLoader(
        train_dataset_image_net, batch_size=batch_size, shuffle=True, num_workers=n_worker,
        drop_last=True
    )
    val_dataloader_image_net = DataLoader(
        val_dataset_image_net,
        batch_size=batch_size,
        shuffle=False, num_workers=n_worker,
        drop_last=True
    )

    return train_dataloader_image_net, val_dataloader_image_net, 5