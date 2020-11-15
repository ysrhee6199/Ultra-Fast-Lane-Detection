import os

import torch
from tqdm import tqdm

from runtime.data.dataset import LaneDataset
from utils.global_config import cfg, adv_cfg


# load images frame by frame and passes them to process_frame
# split: absolute path to test split file, defaults to first entry of cfg.test_splits
# data_root: root directory of dataset, defaults to cfg.data_root
# TODO: no multi-split support here: implement "around" output.py or something like that
def input_images(process_frames, input_file=os.path.join(cfg.data_root, cfg.test_txt), data_root=cfg.data_root):
    dataset = LaneDataset(data_root, input_file, adv_cfg.img_transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

    for i, data in enumerate(tqdm(loader)):
        # for data in loader:
        imgs, names = data
        process_frames(imgs, names)
