# supports multiple input modules
# images: image dataset from a text file containing a list of paths to images
# video: video input
# stream: eg webcam
# supports multiple output modules
# video - generates an output video; same as demo.py
# test - compares result with labels; same as test.py
# prod - will probably only log results, in future you will put your production code here
# i will not provide multi-gpu support. See test.py as a reference
# but it might be more complicated in the end as i dont plan to support multi gpu in any way
#
# A note on performance: This code should provide acceptable performance, but it was not developed with the target of
# achieving the best performance.
# The main goal is to provide a good understandable and expandable / adaptable code base.

import os
import time
from typing import Sequence, List

import torch
from numpy import ndarray

from runtime.input_modules.input_images import input_images
from model.model import parsingNet
from runtime.input_modules.input_screencap import input_screencap
from runtime.input_modules.input_video import input_video
from runtime.out_modules.out_json import JsonOut
from runtime.out_modules.out_prod import ProdOut
from runtime.out_modules.out_test import TestOut
from runtime.out_modules.out_video import VisualOut
from utils.global_config import cfg, adv_cfg


def setup_net():
    """
    setup neural network
    load config and net (from hdd)
    Returns: neural network (torch.nn.Module)
    """
    assert cfg.backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']

    # basic inet of nn
    torch.backends.cudnn.benchmark = True  # automatically select best algorithms
    net = parsingNet(
        pretrained=False,
        backbone=cfg.backbone,
        cls_dim=(cfg.griding_num + 1, adv_cfg.cls_num_per_lane, cfg.num_lanes),
        use_aux=False
    ).cuda()
    # It should be noted that our method only uses the auxiliary segmentation task in the training phase, and it would
    # be removed in the testing phase. In this way, even we added the extra segmentation task, the running speed of our
    # method would not be affected.
    net.eval()

    # load and apply our trained model
    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v
    net.load_state_dict(compatible_state_dict, strict=False)

    return net


def setup_input(process_frame):
    """
    setup data input (where the frames come from)
    Args:
        process_frame: function taking list of frames and a corresponding list of filenames
    """
    if cfg.input_mode == 'images':
        input_images(process_frame, os.path.join(cfg.data_root, cfg.test_txt), cfg.data_root)
    elif cfg.input_mode == 'video':
        input_video(process_frame, '/home/markus/PycharmProjects/datasets/Trainingsdatensatz1/test_vid.mp4',
                    os.path.join(cfg.data_root, cfg.test_txt))
    elif cfg.input_mode == 'camera':
        input_video(process_frame, 0)
    elif cfg.input_mode == 'screencap':
        input_screencap(process_frame, {'top': 0, 'left': 3440, 'width': 1920, 'height': 1080})
    else:
        print(cfg.input_mode)
        raise NotImplemented('unknown/unsupported input_mode')


def setup_out_method():
    """
    setup the output method
    Returns: method/function reference to a function taking
    - a list of predictions
    - a list of corresponding filenames (if available)
    - a list of source_frames (if available)
    """
    if cfg.output_mode == 'video':
        video_out = VisualOut()
        return video_out.out, lambda: None
    elif cfg.output_mode == 'test':
        test_out = TestOut()
        return test_out.out, test_out.post
    elif cfg.output_mode == 'json':
        return JsonOut().out, lambda: None
    elif cfg.output_mode == 'prod':
        prod_out = ProdOut()
        return prod_out.out, prod_out.post
    else:
        print(cfg.output_mode)
        raise NotImplemented('unknown/unsupported output_mode')


class FrameProcessor:
    """
    helper class to process frame
    provides simplified access to process_frame() method
    or a better encapsulation compared to functional approach (depending on implementation ;))
    """

    def __init__(self, net, output_method):
        self.net = net
        self.output_method = output_method
        self.measure_time = True  # TODO: move to cfg
        if self.measure_time:
            self.timestamp = time.time()
            self.avg_fps = []

    def process_frames(self, frames: torch.Tensor, names: List[str] = None, source_frames: List[ndarray] = None):
        """
        process frames and pass result to output_method
        Args:
            frames: frames to process, have to be preprocessed (scaled, as tensor, normalized)
            names: file paths - provide if possible
            source_frames: source images (unscaled, eg from camera) - provide if possible
        """
        if self.measure_time: time1 = time.time()
        with torch.no_grad():
            y = self.net(frames.cuda())  # TODO: maybe use "with torch.no_grad():" to reduce memory usage
        if self.measure_time: time2 = time.time()
        self.output_method(y, names, source_frames)

        if self.measure_time:
            real_time = (time.time() - self.timestamp) / len(y)
            synthetic_time = (time2 - time1) / len(y)
            real_time_wo_out = (time2 - self.timestamp) / len(y)
            print(
                f'fps real: {round(1 / real_time)}, real wo out: {round(1 / real_time_wo_out)}, synthetic: {round(1 / synthetic_time)}, frametime real: {real_time}, real wo out: {real_time_wo_out}, synthetic: {synthetic_time}',
                flush=True)
            self.avg_fps.append((real_time, real_time_wo_out, synthetic_time))
            self.timestamp = time.time()

    def __del__(self):
        if self.measure_time:
            print([round(1 / (sum(y) / len(y))) for y in zip(*self.avg_fps)])


if __name__ == "__main__":
    out_method, post_method = setup_out_method()
    net = setup_net()
    frame_processor = FrameProcessor(net, out_method)
    setup_input(frame_processor.process_frames)
    post_method()  # called when input method is finished (post processing)
