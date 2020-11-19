import json
import os

import cv2


def get_lane_color(i):
    lane_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    return lane_colors[i % 5]


def input_images(input_file, data_root):
    with open(os.path.join(data_root, input_file)) as file:
        lines = file.readlines()

    for line in lines:
        dict = json.loads(line)

        image = cv2.imread(os.path.join(data_root, dict['raw_file']))

        for i in range(len(dict['lanes'])):
            lane = dict['lanes'][i]
            for j in range(len(dict['h_samples'])):
                # if lane[j] is not -2:
                    cv2.circle(image, (lane[j], dict['h_samples'][j]), 5, get_lane_color(i), -1)

        cv2.imshow('video', image)
        cv2.waitKey(1)


if __name__ == '__main__':
    input_images('test.json', '/home/markus/OneDrive/Projekt_-_Fast_Lane_Detection/Datensätze/Datensatz03_v3/')