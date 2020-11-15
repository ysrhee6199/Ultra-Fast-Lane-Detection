from itertools import count

import cv2
from PIL import Image

from data.constant import default_img_transforms


def input_video(process_frames, input_file, names_file):
    """
    read a video file
    batch size is always 1
    Args:
        process_frames: function taking a list of preprocessed frames, file paths and source frames
        input_file: video file
        names_file: list with file paths to the frames of the video

    Returns:

    """
    if not input_file:
        raise Exception('input file required')
    if not names_file:
        print('no names_file specified, some functions (output modules) might not work as they require names!')
    else:
        with open(names_file, 'r') as file:
            image_paths = file.read().splitlines()

    vid = cv2.VideoCapture(input_file)

    for i in count():
        success, image = vid.read()
        if not success: break

        frame = Image.fromarray(image)
        # unsqueeze: adds one dimension to tensor array (to be similar to loading multiple images)
        frame = default_img_transforms(frame).unsqueeze(0)

        process_frames(frame, [image_paths[i]] if names_file else None, [image])
