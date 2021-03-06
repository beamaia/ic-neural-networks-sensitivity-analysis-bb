import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms

from glob import glob
import cv2
import sys

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.image import PatchExtractor
import numpy as np

################ Dataset path in Turing server ################
TURING_train = '/home/turing/Desktop/humberto/Dataset-Dividido-aux/train/'
TURING_val =  '/home/turing/Desktop/humberto/Dataset-Dividido-aux/val/'
TURING_test = '/home/turing/Desktop/humberto/Dataset-Dividido-aux/test/'

################ Dataset path in Hinton server ################
HINTON_train = '/home/hinton/Desktop/Dados/Dataset-Dividido-aux/train/'
HINTON_val = '/home/hinton/Desktop/Dados/Dataset-Dividido-aux/val/'
HINTON_test = '/home/hinton/Desktop/Dados/Dataset-Dividido-aux/test/'

############### DATA AUGMENTATION

DATA_WITH_AUG = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(45),
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(45),
            transforms.ToTensor()
        ]),
        'test': transforms.Compose([
            transforms.ToTensor()
        ])
}

############### NO DATA AUGMENTATION

DATA_NO_AUG = {
        'train': transforms.Compose([
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.ToTensor()
        ]),
        'test': transforms.Compose([
            transforms.ToTensor()
        ])
}

def image_paths():
    print("Getting images' path...\n")

    PATH_train = HINTON_train
    PATH_val = HINTON_val
    PATH_test = HINTON_test
    
    normal = 'classe0(normal)/*'
    carcinoma = 'classe1(carcinoma)/*'

    train_normal = glob(PATH_train + normal)
    train_carcinoma = glob(PATH_train + carcinoma)
    
    val_normal = glob(PATH_val + normal)
    val_carcinoma = glob(PATH_val + carcinoma)

    test_normal = glob(PATH_test + normal)
    test_carcinoma = glob(PATH_test + carcinoma)

    return train_normal, train_carcinoma, val_normal, val_carcinoma, test_normal, test_carcinoma

def process_images(images, height=224, width=224):
    print("Processing images...", end="\n\n")
    
    pros_images = []
        
    for img in images:
        full_size_image = cv2.imread(img)
        image = (cv2.resize(full_size_image, (width, height), interpolation = cv2.INTER_CUBIC))
        
        pros_images.append(image)

    pros_images = np.array(pros_images)

    return pros_images

def create_images_labels(x_normal, x_carcinoma):

    print("Creating images and labels...")
    images = np.concatenate((x_normal, x_carcinoma), axis=0)

    labels_nr = np.zeros(len(x_normal))
    labels_ca = np.ones(len(x_carcinoma))
    labels = np.concatenate((labels_nr, labels_ca))

    images = torch.from_numpy(images)
    labels = torch.Tensor(labels) 

    images = images.type(torch.FloatTensor) 
    labels = labels.type(torch.LongTensor) 
    
    return images, labels

def create_train_val_test(hw):
    print("Creating train, validation and test images...", end="\n\n")

    xn_train_paths, xc_train_paths, xn_val_paths, xc_val_paths, xn_test_paths, xc_test_paths = image_paths()

    xn_train = process_images(xn_train_paths, hw, hw)
    xc_train = process_images(xc_train_paths, hw, hw)
    xn_val = process_images(xn_val_paths, hw, hw)
    xc_val = process_images(xc_val_paths, hw, hw)
    xn_test = process_images(xn_test_paths, hw, hw)
    xc_test = process_images(xc_test_paths, hw, hw)

    normal_len = len(xn_train) + len(xn_val) + len(xn_test)
    carcinoma_len = len(xc_train) + len(xc_val) + len(xc_test)

    x_train, y_train = create_images_labels(xn_train, xc_train)
    x_val, y_val = create_images_labels(xn_val, xc_val)
    x_test, y_test = create_images_labels(xn_test, xc_test)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test), (normal_len, carcinoma_len)

# DataLoader

class DatasetOral(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform):
        self.labels = labels
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

def create_dataloaders(x_train, y_train, x_val, y_val, x_test, y_test, train_sample_weights, batch_size=64, shuffle=True, num_workers=2):
    print("Creating dataloaders...", end='\n\n')

    data_transforms = DATA_NO_AUG

    params = {'batch_size': batch_size,
              'num_workers': num_workers}

    x_train /=255.
    x_val /=255.
    x_test /= 255.

    print("Creating samplers...")
    
    train_sampler = WeightedRandomSampler(train_sample_weights, len(train_sample_weights))
    
    w, h = x_train.shape[1], x_train.shape[2]

    x_train = x_train.reshape(-1, 3, w, h)
    x_val = x_val.reshape(-1, 3, w, h)
    x_test = x_test.reshape(-1, 3, w, h)

    # print(x_train.shape, x_val.shape, x_test.shape, sep="\n")

    train_set = DatasetOral(x_train, y_train, data_transforms['train'])
    train_loader = DataLoader(train_set, sampler=train_sampler, **params)

    val_set = DatasetOral(x_val, y_val, data_transforms['val'])
    val_loader = DataLoader(val_set, **params)

    test_set = DatasetOral(x_test, y_test, data_transforms['test'])
    test_loader = DataLoader(test_set, **params)

    return train_loader, val_loader, test_loader

