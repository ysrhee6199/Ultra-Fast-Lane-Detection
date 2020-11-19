from itertools import count

import cv2
import numpy as np
import typing
from PIL import Image
from mss import mss

from utils.global_config import adv_cfg


def input_screencap(process_frames: typing.Callable, mon: dict) -> None:
    """
    record from screen
    batch size is always 1

    This is was implemented to test GTA. Its a bit difficult to use. You have to manually specify the
    position and size of your target window here. If your information are wrong (out of screen) you'll get
    a cryptic exception!
    Make sure your config resolution matches your settings here.
    TODO: scale source image to config settings

    Args:
        process_frames: function taking a list of preprocessed frames, file paths and source frames
        mon: position and size of recording window, eg {'top': 0, 'left': 3440, 'width': 1920, 'height': 1080}
    """

    sct = mss()

    for i in count():
        screenshot = sct.grab(mon)
        image = Image.frombytes("RGB", (screenshot.width, screenshot.height), screenshot.rgb)

        # unsqueeze: adds one dimension to tensor array (to be similar to loading multiple images)
        frame = adv_cfg.img_transform(image).unsqueeze(0)

        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        process_frames(frame, None, [image])
