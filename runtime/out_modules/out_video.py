import os

import cv2
import numpy as np
import scipy

from utils.global_config import cfg


def map_x_to_image(y):
    """
    Map x estimations to image coordinates
    Args:
        y: one result sample (can be directly from net or post-processed -> all number types should be accepted)

    Returns: x coordinates for each lane
    """
    lanes = []
    offset = 0.5  # different values used in ufld project. demo: 0.0, test: 0.5

    for i in range(y.shape[1]):
        out_i = y[:, i]
        lane = [
            int((loc + offset) * float(cfg.img_width) / (cfg.griding_num - 1))
            # int(round((loc + 0.5) * float(cfg.img_width) / (cfg.griding_num - 1)))
            if loc != -2
            else -2
            for loc in out_i
        ]
        lanes.append(lane)
    return lanes


def improve_x_vals(y):
    """
    Creates more accurate x values.
    Tries to improve the estimation by including all probabilities instead of only using the most probable class
    Args:
        y: one result sample

    Returns:
        2D array containing x values (float) per h_sample and lane
    """
    out = y.data.cpu().numpy()  # load data to cpu and convert to numpy
    out_loc = np.argmax(out, axis=0)  # get most probably x-class per lane and h_sample

    # do some stuff i dont fully understand to improve x accuracy
    prob = scipy.special.softmax(out[:-1, :, :], axis=0)  # relative probability with sum() == 1.0
    idx = np.arange(cfg.griding_num).reshape(-1, 1, 1)  # init 3 dim array containing numbers from 0 to griding_num - 1
    loc = np.sum(prob * idx, axis=0)  # calculate more accurate x values

    loc[out_loc == cfg.griding_num] = -2  # where the most probable class is 100 (no lane detected): replace with -2
    return loc


lane_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]


class VideoOut:
    def __init__(self, dataset_filename):
        # init video out
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out_filename = dataset_filename[:-3] + 'avi'
        self.vout = cv2.VideoWriter(out_filename[:-3] + 'avi', fourcc, 30.0, (cfg.img_width, cfg.img_height))

    def out_video(self, y, names):
        """
        Generate video output
        Args:
            y: network result
            names: filenames for y
        """
        for i in range(0, len(y)):
            lanes = np.array(map_x_to_image(improve_x_vals(y[i])))

            vis = cv2.imread(os.path.join(cfg.data_root, names[i]))
            for i in range(lanes.shape[0]):  # iterate over lanes
                lane = lanes[i, :]
                if np.sum(lane != -2) > 2:  # If more than two points found for this lane
                    color = lane_colors[i]
                    for j in range(lanes.shape[1]):
                        img_x = lane[j]
                        img_y = cfg.h_samples[j]
                        if img_x != -2:
                            cv2.circle(vis, (img_x, img_y), 5, color, -1)
            cv2.imshow('video', vis)
            cv2.waitKey(1)
            self.vout.write(vis)
