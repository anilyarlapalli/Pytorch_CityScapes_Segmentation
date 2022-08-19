import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import warnings
warnings.filterwarnings('ignore')
import segmentation_models_pytorch as seg_models

names = ["PSPNet", "UNet", "Unet++", "FPN", "DeepLab V3", "DeepLab V3+"]
# models_dict = {
#     "PSPNet": seg_models.PSPNet(classes=3),
#     "UNet": seg_models.Unet(classes=3),
#     "Unet++": seg_models.UnetPlusPlus(classes=3),
#     "FPN": seg_models.FPN(classes=3),
#     "DeepLab V3": seg_models.DeepLabV3(classes=3),
#     "DeepLab V3+": seg_models.DeepLabV3Plus(classes=3),
# }

models_dict = { "Unet++": seg_models.UnetPlusPlus(classes=3),}
