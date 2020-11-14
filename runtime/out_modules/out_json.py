import json
import os

import numpy as np

from runtime.out_modules.common import get_filename_date_string, map_x_to_image, evaluate_predictions
from utils.global_config import cfg


class JsonOut:
    """
    provides the ability to different visual output types
    - live video
    - record video
    - save images
    visualization can be points or lines
    """

    def __init__(
            self,
            filepath=os.path.join(
                cfg.log_path,
                f'{get_filename_date_string()}_{cfg.dataset}_{os.path.splitext(cfg.test_txt)[0]}.json'
            )
    ):
        """
        Args:
            filepath: full file path where the results will be stored
        """
        self.filepath = filepath
        self.out_file = open(self.filepath, 'w')

    def out(self, y, names):
        """
        Generate json output to text file
        Args:
            y: network result (list of samples)
            names: filenames for y
        """
        # iterate over samples
        for i in range(len(y)):
            lanes = np.array(map_x_to_image(evaluate_predictions(y[i])))  # get x coordinates based on probabilities

            json_string = json.dumps({
                'lanes': lanes,
                'h_samples': cfg.h_samples,
                'raw_file': names[i]
            })

            self.out_file.write(json_string + '\n')
