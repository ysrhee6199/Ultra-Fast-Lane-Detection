import os
import time
from typing import Tuple, List

import cv2
import numpy as np
import torch

from runtime.out_modules.common import get_filename_date_string, map_x_to_image, evaluate_predictions
from utils.global_config import cfg, adv_cfg


def get_lane_color(i: int) -> Tuple:
    """ Get a predefined colors depending on i. Colors repeat if i gets to big

    Args:
        i: any number, same number, same color

    Returns: Tuple containing 3 values, eg (255, 0, 0)
    """
    lane_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    return lane_colors[i % 5]


class VisualOut:
    """
    provides the ability to different visual output types

    * live video
    * record video
    * save images

    visualization can be points or lines
    """

    def __init__(
            self,
            enable_live_video=cfg.video_out_enable_live_video,
            enable_video_export=cfg.video_out_enable_video_export,
            enable_image_export=cfg.video_out_enable_image_export,
            enable_line_mode=cfg.video_out_enable_line_mode,
    ):
        """
        used non-basic-cfg values: cfg.video_out_enable_live_video, cfg.video_out_enable_video_export, cfg.video_out_enable_image_export, cfg.video_out_enable_line_mode

        Args:
            enable_live_video: show video
            enable_video_export: save as video to disk
            enable_image_export: save as image files to disk
            enable_line_mode: visualization as lines instead of dots
        """
        self.enable_live_video = enable_live_video
        self.enable_video_export = enable_video_export
        self.enable_image_export = enable_image_export
        self.enable_line_mode = enable_line_mode

        if enable_video_export:
            # init video out
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out_filename = f'{get_filename_date_string()}_{cfg.dataset}.avi'
            out_full_path = os.path.join(cfg.log_path, out_filename)
            print(out_full_path)
            self.vout = cv2.VideoWriter(out_full_path, fourcc, 30.0, (cfg.img_width, cfg.img_height))

    def out(self, y: torch.Tensor, names: List[str], frames: List[np.ndarray]):
        """ Generate visual output

        Args:
            y: network result (list of samples containing probabilities per sample)
            names: filenames for y, if empty: frames have to be provided
            frames: source frames, if empty: names have to be provided
        """
        if not names and not frames:
            raise Exception('at least frames or names have to be provided')
        # iterate over samples
        for i in range(len(y)):
            lanes = np.array(map_x_to_image(evaluate_predictions(y[i])))  # get x coordinates based on probabilities

            if frames:
                vis = frames[i]
            else:
                vis = cv2.imread(os.path.join(cfg.data_root, names[i]))

            if vis is None:
                raise Exception('failed to load frame')

            for i in range(lanes.shape[0]):  # iterate over lanes
                lane = lanes[i, :]
                if np.sum(lane != -2) > 2:  # If more than two points found for this lane
                    color = get_lane_color(i)
                    for j in range(lanes.shape[1]):
                        img_x = lane[j]
                        img_y = adv_cfg.scaled_h_samples[j]
                        if img_x != -2:
                            if self.enable_line_mode:
                                if j > 0:
                                    cv2.line(vis, (lane[j - 1], adv_cfg.scaled_h_samples[j - 1]), (img_x, img_y), color,
                                             5)
                            else:
                                cv2.circle(vis, (img_x, img_y), 5, color, -1)
            if self.enable_live_video:
                cv2.imshow('video', vis)
                cv2.waitKey(1)
            if self.enable_video_export:
                self.vout.write(vis)
            if self.enable_image_export:
                out_path = os.path.join(
                    cfg.log_path,
                    f'{get_filename_date_string()}_out', names[i] if names else int(time.time() * 1000000)
                )  # use current timestamp (nanoseconds) as fallback
                cv2.imwrite(out_path, vis)
