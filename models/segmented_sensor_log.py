import os

import matplotlib.pyplot as plt
import seaborn as sn
import unicodecsv as csv

from models.topological_compat_matrix import TopologicalCompatMatrix
from sequence_classification.sequence_classifier_input import SequenceClassifierInput
from utils.constants import DATA_FOLDER, LOG_ENTRY_DELIMITER, SENSOR_ID_POS


class SegmentedSensorLog(object):
    def __init__(self, sensor_log=None, top_compat_matrix=None, threshold=None, segments=None,
                 sensor_id_pos=SENSOR_ID_POS):
        """
        Build segmented version of the given log considering the given probabilistic topological compatibility matrix.
        
        :type sensor_log: file
        :type top_compat_matrix: TopologicalCompatMatrix
        :type threshold: float
        :param sensor_log: the tab-separated file containing the sensor log.
        :param top_compat_matrix: the topological compatibility matrix of the sensor log.
        :param threshold: the threshold to reach for a direct succession to be significant.
        :param segments: a precomputed list of segments. 
        :param sensor_id_pos: the position of the sensor id in the log entry. 
        """
        if segments:
            self.segments = segments

        elif sensor_log and top_compat_matrix and threshold:
            self.segments = []
            self.sensor_log = csv.reader(sensor_log, delimiter=LOG_ENTRY_DELIMITER)
            self.top_compat_matrix = top_compat_matrix

            s0 = next(self.sensor_log, None)  # consider a sliding window of two events per step
            s1 = next(self.sensor_log, None)
            segment = [list(s0)]
            while s0 is not None and s1 is not None:
                s0_id = s0[sensor_id_pos]
                s1_id = s1[sensor_id_pos]

                if self.top_compat_matrix.prob_matrix[s0_id][s1_id] >= threshold:
                    # the direct succession value is above the threshold
                    segment.append(list(s1))  # continue the segment
                elif segment:
                    # the direct succession value is under the threshold (and the current segment is non-empty)
                    self.segments.append(list(segment))  # store a copy of the segment so far
                    segment = [list(s1)]                 # start the new segment from the second item in the window

                # prepare next step (slide the window by one position)
                s0 = s1
                s1 = next(self.sensor_log, None)

        else:
            raise ValueError('Not enough inputs provided.')

    def plot_stats(self, distribution=True, time_series=True):
        if not (distribution or time_series):
            raise ValueError('At least a chart should be plotted.')

        segments_num = len(self.segments)
        segments_lengths = [len(s) for s in self.segments]

        print('segments num:', segments_num)
        print('min length:  ', len(min(self.segments, key=len)))
        print('max length:  ', len(max(self.segments, key=len)))
        print('avg length:  ', sum(segments_lengths) / segments_num)

        if distribution:
            plt.figure()
            sn.distplot(segments_lengths)

        if time_series:
            plt.figure()
            sn.tsplot(segments_lengths)

        plt.show()


if __name__ == '__main__':
    # import pickle
    # from pprint import pprint

    # SRC = os.path.join(DATA_FOLDER, 'sequences.pkl')
    #
    # with open(SRC, 'rb') as src:
    #     segs = dict(pickle.load(src))
    #
    # ssl = SegmentedSensorLog(segments=segs.keys())

    SRC = os.path.join(DATA_FOLDER, 'complete_dataset_preprocessed_filtered_simplified.txt')
    THRESHOLD = 0.1

    with open(SRC, 'rb') as log:
        tcm = TopologicalCompatMatrix(log, sensor_id_pos=0)

    with open(SRC, 'rb') as log:
        ssl = SegmentedSensorLog(log, tcm, THRESHOLD, sensor_id_pos=0)

    GOOD = 'GOOD'
    sequences = []
    labels = []
    for segment_ in ssl.segments:
        if len(segment_) > 1:
            sequence = ''
            for c in segment_:
                sequence += c[0]
            sequences.append(sequence)
            labels.append(GOOD)

    clf_input = SequenceClassifierInput(sequences=sequences, labels=labels)
    clf_input.get_spectrum_train_test_data()
