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

import torch

from runtime.input_modules.input_images import input_images
from model.model import parsingNet
from runtime.out_modules.out_json import JsonOut
from runtime.out_modules.out_prod import ProdOut
from runtime.out_modules.out_test import out_test
from runtime.out_modules.out_video import ImageOut
from utils.global_config import cfg


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
        cls_dim=(cfg.griding_num + 1, cfg.cls_num_per_lane, cfg.num_lanes),
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
        raise NotImplemented
    elif cfg.input_mode == 'stream':
        raise NotImplemented
    else:
        print(cfg.input_mode)
        raise NotImplemented('unknown/unsupported input_mode')


def setup_out_method():
    """
    setup the output method
    Returns: method/function reference to a function taking a list of predictions and a list of corresponding filenames
    """
    if cfg.output_mode == 'video':
        video_out = ImageOut()
        return video_out.out
    elif cfg.output_mode == 'test':
        return out_test
    elif cfg.output_mode == 'json':
        return JsonOut().out
    elif cfg.output_mode == 'prod':
        return ProdOut().out
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

    def process_frame(self, frames, names):
        """
        process frames and pass result to output_method
        Args:
            frames: frames to process
            names: file paths, for output_method
        """
        y = self.net(frames.cuda())  # TODO: maybe use "with torch.no_grad():" to reduce memory usage
        self.output_method(y, names)


if __name__ == "__main__":
    out_method = setup_out_method()
    net = setup_net()
    frame_processor = FrameProcessor(net, out_method)
    setup_input(frame_processor.process_frame)
