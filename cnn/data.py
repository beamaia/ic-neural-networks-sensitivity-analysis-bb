import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms

from glob import glob
import cv2
import sys

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.image import PatchExtractor
import numpy as np

def image_paths(set_num=1, server="bea"):
    print("Getting images' path...")
    print(f"Server: {server} - Set: {set_num}")
    first_set= set_num == 1
    both_sets = set_num == 3

    home = server.lower() == "bea"
    hinton = server.lower() == "hinton"
    turing = server.lower() == "turing"

    # EDIT THIS

    PATH_normal100x = 'First Set/100x Normal Oral Cavity Histopathological Images/*'
    PATH_carcinoma100x = 'First Set/100x OSCC Histopathological Images/*'
    PATH_normal400x = 'Second Set/400x Normal Oral Cavity Histopathological Images/*'
    PATH_carcinoma400x = 'Second Set/400x OSCC Histopathological Images/*'
    
    if home:
        PATH_ini = ".././data/oralCancer-Borooah/"
    elif hinton:
        PATH_ini = "/home/hinton/Desktop/Humberto/"
    elif turing:
        PATH_ini = ".././oralCancer-Borooah/"
    else:
        print("Server not identified")
        sys.exit(1)
        
    PATH_normal100 = PATH_ini + PATH_normal100x
    PATH_carcinoma100 = PATH_ini + PATH_carcinoma100x
    PATH_normal400 = PATH_ini + PATH_normal400x
    PATH_carcinoma400 = PATH_ini + PATH_carcinoma400x
    if not both_sets:
        if first_set:
            images_normal = glob(PATH_normal100)
            images_carcinoma = glob(PATH_carcinoma100)
        else:
            images_normal = glob(PATH_normal400)
            images_carcinoma = glob(PATH_carcinoma400)      

    else:
        images_normal100 = glob(PATH_normal100)
        images_carcinoma100 = glob(PATH_carcinoma100)
        images_normal400 = glob(PATH_normal400)
        images_carcinoma400 = glob(PATH_carcinoma400)

        images_normal = images_normal100 + images_normal400
        images_carcinoma = images_carcinoma100 + images_carcinoma400

    print(f"Found {len(images_normal)} images of class: normal and {len(images_carcinoma)} images of class: carcinoma", end="\n\n")
    
    if len(images_normal) == 0 or len(images_carcinoma) == 0:
        print("Folder not found!")
        sys.exit(1)
    return images_normal, images_carcinoma

def process_images(images, height=1536, width=2048):
    print("Processing images...", end="\n\n")
    
    pros_images = []
        
    for img in images:
        full_size_image = cv2.imread(img)
        image = (cv2.resize(full_size_image, (width, height), interpolation = cv2.INTER_CUBIC))
        
        pros_images.append(image)

    pros_images = np.array(pros_images)

    return pros_images

def create_images_labels(x_normal, x_carcinoma, patch_size=224, max_patches=30):

    print("Creating patches...")
    
    # pe = PatchExtractor(patch_size = (patch_size, patch_size), max_patches = max_patches)
    # patches_normal = pe.transform(x_normal)
    # patches_carcinoma = pe.transform(x_carcinoma)
    # images = np.concatenate((patches_normal, patches_carcinoma), axis = 0)

    # labels_nr = np.zeros(len(patches_normal))
    # labels_ca = np.ones(len(patches_carcinoma))
    # labels = np.concatenate((labels_nr, labels_ca))

    images = np.concatenate((x_normal, x_carcinoma), axis=0)

    labels_nr = np.zeros(len(x_normal))
    labels_ca = np.ones(len(x_carcinoma))
    labels = np.concatenate((labels_nr, labels_ca))

    images = torch.from_numpy(images)
    labels = torch.Tensor(labels) 

    images = images.type(torch.FloatTensor) 
    labels = labels.type(torch.LongTensor) 
    
    print("Created images and labels...")
    return images, labels

def create_train_test(images, labels, test_size):
    print("Creating train, validation and test images...", end="\n\n")

    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, random_state=11)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size =0.5, random_state=11)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

# DataLoader

class DatasetOral(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.labels = labels
        self.images = images

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

def create_dataloaders(x_train, y_train, x_val, y_val, x_test, y_test, train_sample_weights, val_sample_weights, test_sample_weights, batch_size=64, shuffle=True, num_workers=2):
    print("Creating dataloaders...", end='\n\n')
    params = {'batch_size': batch_size,
              'num_workers': num_workers}

    x_train /=255.
    x_val /=255.
    x_test /= 255.

    print("Creating samplers...")
    
    train_sampler = WeightedRandomSampler(train_sample_weights, len(train_sample_weights))
    # val_sampler = WeightedRandomSampler(val_sample_weights, len(val_sample_weights))
    # test_sampler = WeightedRandomSampler(test_sample_weights, len(test_sample_weights))

    w, h = x_train.shape[1], x_train.shape[2]
    x_train = x_train.reshape(-1, 3, w, h)
    x_val = x_val.reshape(-1, 3, w, h)
    x_test = x_test.reshape(-1, 3, w, h)

    # print(x_train.shape, x_val.shape, x_test.shape, sep="\n")

    train_set = DatasetOral(x_train, y_train)
    train_loader = DataLoader(train_set, sampler=train_sampler, **params)

    val_set = DatasetOral(x_val, y_val)
    val_loader = DataLoader(val_set, **params)

    test_set = DatasetOral(x_test, y_test)
    test_loader = DataLoader(test_set, **params)

    return train_loader, val_loader, test_loader

