# load pytorch dataset
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.nn as nn

import utils

#Custom Segmentation class dataset
class CityScapesSegDataset(torch.utils.data.Dataset):
    """A customized dataset to load the VOC dataset."""
    def __init__(self, files, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = utils.read_voc_images(voc_dir, files)
        self.features = [self.normalize_image(feature) for feature in self.filter(features)]
        self.labels = [self.normalize_image(label) for label in self.filter(labels)]
        print('read ' + str(len(self.features)) + ' examples')
        
    def normalize_image(self, img):
        return self.transform(img.float() / 255)
    
    def filter(self, imgs):
        return [img for img in imgs if (img.shape[1] >= self.crop_size[0] and img.shape[2] >= self.crop_size[1])]
    
    def __getitem__(self, idx):
        feature, label = utils.voc_rand_crop(self.features[idx], self.labels[idx], *self.crop_size)
        return (feature, label)

    def __len__(self):
        return len(self.features)



