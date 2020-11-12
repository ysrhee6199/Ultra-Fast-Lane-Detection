# row anchors are a series of pre-defined coordinates in image height to detect lanes
# the row anchors are defined according to the evaluation protocol of CULane and Tusimple
# since our method will resize the image to 288x800 for training, the row anchors are defined with the height of 288
# you can modify these row anchors according to your training image resolution
from threading import Thread

from PIL import Image
from torchvision import transforms

from utils import global_config


def gen_row_anchor():
    return [x * global_config.cfg.train_img_height / global_config.cfg.img_height for x in global_config.cfg.h_samples]


default_img_transforms = transforms.Compose([
    transforms.Resize((global_config.cfg.train_img_height, global_config.cfg.train_img_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


