
import os
import torch
import torchvision

def read_voc_images(voc_dir, files):
    """Read all VOC feature and label images."""

    mode = torchvision.io.image.ImageReadMode.RGB

    features, labels = [], []
    for i, fname in enumerate(files):
        image = torchvision.io.read_image(os.path.join(voc_dir, fname), mode)
        feature = image[:, :, :256]
        label = image[:, :, 256:]
        features.append(feature)
        labels.append(label)

    return(features, labels)

def voc_rand_crop(feature, label, height, width):
    # print("Randomly crop both feature and label images")
    # print(feature.shape, label.shape)
    rect = torchvision.transforms.RandomCrop.get_params(feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label

def dice(pred, label):
    pred = (pred > 0).float()
    return 2. * (pred*label).sum() / (pred+label).sum()