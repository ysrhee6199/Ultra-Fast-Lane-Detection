import numpy as np
from sklearn.linear_model import LinearRegression
import json


class LaneEval(object):
    lr = LinearRegression()
    pixel_thresh = 20
    pt_thresh = 0.85

    @staticmethod
    def get_angle(xs, y_samples):
        xs, ys = xs[xs >= 0], y_samples[xs >= 0]
        if len(xs) > 1:
            LaneEval.lr.fit(ys[:, None], xs)
            k = LaneEval.lr.coef_[0]
            theta = np.arctan(k)
        else:
            theta = 0
        return theta

    @staticmethod
    def line_accuracy(pred, gt, thresh):
        pred = np.array([p if p >= 0 else -100 for p in pred])
        gt = np.array([g if g >= 0 else -100 for g in gt])
        return np.sum(np.where(np.abs(pred - gt) < thresh, 1., 0.)) / len(gt)

    @staticmethod
    def bench(pred, gt, y_samples):
        """ bench one sample

        Args:
            pred: predicted lanes
            gt: validation lanes
            y_samples: y coordinates

        Returns: accuracy, false positives, false negatives
        """
        if any(len(p) != len(y_samples) for p in pred):  # validate correct number of y_samples
            raise Exception('Format of lanes error.')

        if len(gt) + 2 < len(pred):
            return 0., 0., 1.

        # dont know what this is exactly for
        angles = [LaneEval.get_angle(np.array(x_gts), np.array(y_samples)) for x_gts in gt]
        threshs = [LaneEval.pixel_thresh / np.cos(angle) for angle in angles]

        # init loop vars
        line_accs = []
        fp, fn = 0., 0.
        matched = 0.

        for x_gts, thresh in zip(gt, threshs):
            # determine accuracy for each lane. Each point of each lane can either have an accuracy of 1 or 0
            # depending on whether its distance from the reference/validation point is greater than thresh or not
            # now i'm guessing a bit (didnt analyze the code far enough to be sure):
            # angles contains the tilt of each lane. threshs equals the distance in pixels from which on a point
            # would be counted as a point of another lane.
            # Following that assumptions the accuracy doesn't really tell us how exact a lane is predicted
            # its more like "i'm that sure that i can SEPARATE lane markings"
            # UPDATE: seems to not be (completely) right as the trhehs calculations contains a "pixel_thresh"
            accs = [LaneEval.line_accuracy(np.array(x_preds), np.array(x_gts), thresh) for x_preds in pred]
            max_acc = np.max(accs) if len(accs) > 0 else 0.

            # counting false positives/negatives depending on whether a lines accuracy is greater or smaller than
            # pt_thresh
            if max_acc < LaneEval.pt_thresh:
                fn += 1
            else:
                matched += 1
            line_accs.append(max_acc)
        fp = len(pred) - matched
        if len(gt) > 4 and fn > 0:
            fn -= 1
        s = sum(line_accs)
        if len(gt) > 4:
            s -= min(line_accs)

        # till now s, fp, fn were summed up so we have to divide them by the amount of lines
        # the divisor looks that complex because the line numbers might vary (at least i think its what thats doing)
        return s / max(min(4.0, len(gt)), 1.), fp / len(pred) if len(pred) > 0 else 0., fn / max(min(len(gt), 4.), 1.)

    @staticmethod
    def bench_one_submit(json_pred, json_gt):
        """ Do bench for a list of json-results by calling bench() for every sample

        Args:
            json_pred: predicted list
            json_gt: compare list

        Returns: bench result
        """
        gts = {line['raw_file']: line for line in json_gt}  # reformat: filename as dict key
        accuracy, fp, fn = 0., 0., 0.
        for pred in json_pred:
            # init some variables and do some basic validation
            if 'raw_file' not in pred or 'lanes' not in pred:
                raise Exception('raw_file or lanes not in some predictions.')
            raw_file = pred['raw_file']
            if raw_file not in gts:
                raise Exception('Some raw_file from your predictions do not exist in the test tasks.')
            gt = gts[raw_file]

            # do bench for current sample
            try:
                a, p, n = LaneEval.bench(pred['lanes'], gt['lanes'], gt['h_samples'])
            except BaseException as e:
                raise Exception('Format of lanes error.')
            accuracy += a
            fp += p
            fn += n
        num = len(gts)
        # the first return parameter is the default ranking parameter
        return json.dumps([
            {'name': 'Accuracy', 'value': accuracy / num, 'order': 'desc'},
            {'name': 'FP', 'value': fp / num, 'order': 'asc'},
            {'name': 'FN', 'value': fn / num, 'order': 'asc'}
        ])
