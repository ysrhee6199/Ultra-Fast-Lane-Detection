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

import time
from typing import List

import torch
import typing
from numpy import ndarray

from src.runtime.modules.input.input_images import input_images
from src.runtime.modules.input.input_screencap import input_screencap
from src.runtime.modules.input.input_video import input_video, input_camera
from src.runtime.modules.output.out_json import JsonOut
from src.runtime.modules.output.out_prod import ProdOut
from src.runtime.modules.output.out_test import TestOut
from src.runtime.modules.output.out_video import VisualOut
from src.utils.model.model import parsingNet

from src.utils.global_config import cfg, adv_cfg


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
    # use_aux: It should be noted that our method only uses the auxiliary segmentation task in the training phase, and it would
    # be removed in the testing phase. In this way, even we added the extra segmentation task, the running speed of our
    # method would not be affected.
    # .eval: set module to evaluation mode
    net.eval()

    # load and apply our trained model
    state_dict = torch.load(cfg.trained_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v
    net.load_state_dict(compatible_state_dict, strict=False)

    return net


def setup_input(process_frames: typing.Callable[[torch.Tensor, typing.List[str], typing.List[ndarray]], None]):
    """ setup data input (where the frames come from)

    Args:
        process_frames: function taking list of frames and a corresponding list of filenames
    """
    if cfg.input_mode == 'images':
        input_images(process_frames)
    elif cfg.input_mode == 'video':
        input_video(process_frames)
    elif cfg.input_mode == 'camera':
        input_camera(process_frames)
    elif cfg.input_mode == 'screencap':
        input_screencap(process_frames,
                        {
                            'top': cfg.screencap_recording_area[1],
                            'left': cfg.screencap_recording_area[0],
                            'width': cfg.screencap_recording_area[2],
                            'height': cfg.screencap_recording_area[3]
                        })
    else:
        print(cfg.input_mode)
        raise NotImplemented('unknown/unsupported input_mode')


def setup_out_method():
    """ setup the output method

    Returns: method/function reference to a function taking

    * a list of predictions
    * a list of corresponding filenames (if available)
    * a list of source_frames (if available)

    """
    if cfg.output_mode == 'video':
        video_out = VisualOut()
        return video_out.out, lambda: None
    elif cfg.output_mode == 'test':
        test_out = TestOut()
        return test_out.out, test_out.post
    elif cfg.output_mode == 'json':
        json_out = JsonOut()
        return json_out.out, json_out.post
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
        self.measure_time = cfg.measure_time
        if self.measure_time:
            self.timestamp = time.time()
            self.avg_fps = []

    def process_frames(self, frames: torch.Tensor, names: List[str] = None, source_frames: List[ndarray] = None):
        """ process frames and pass result to output_method

        Args:
            frames: frames to process, have to be preprocessed (scaled, as tensor, normalized)
            names: file paths - provide if possible
            source_frames: source images (unscaled, eg from camera) - provide if possible
        """
        if self.measure_time: time1 = time.time()
        with torch.no_grad():  # no_grad: disable gradient calculation. Reduces (gpu) memory consumption
            y = self.net(frames.cuda())
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


def main():
    out_method, post_method = setup_out_method()
    net = setup_net()
    frame_processor = FrameProcessor(net, out_method)
    setup_input(frame_processor.process_frames)
    post_method()  # called when input method is finished (post processing)
