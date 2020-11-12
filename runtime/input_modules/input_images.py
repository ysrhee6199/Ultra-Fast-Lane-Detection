import os

import torch
from tqdm import tqdm

from data.constant import gen_row_anchor, default_img_transforms
from model.model import parsingNet
from runtime.data.dataset import LaneDataset
from utils.dist_utils import dist_print
from utils.global_config import cfg


# load images frame by frame and passes them to process_frame
# split: absolute path to test split file, defaults to first entry of cfg.test_splits
# data_root: root directory of dataset, defaults to cfg.data_root
# TODO: no multi-split support here: implement "around" output.py or something like that
def input_images(process_frame, split=None, data_root=cfg.data_root):
    split = split if split else os.path.join(cfg.data_root, cfg.test_splits[0])
    dataset = LaneDataset(data_root, split, default_img_transforms)

    # getting resolution from config, might refactor this and get data from image
    img_w, img_h = cfg.img_width, cfg.img_height

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    # for i, data in enumerate(tqdm(loader)):
    for data in loader:
        #     seems to allow multiple images at the same tame (-> batch_size)
        imgs, names = data
        process_frame(imgs, names)
