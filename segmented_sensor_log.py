import os
from pprint import pprint

import matplotlib.pyplot as plt
import seaborn as sn
import unicodecsv as csv

from topological_compat_matrix import TopologicalCompatMatrix
from utils.constants import DATA_FOLDER, LOG_ENTRY_DELIMITER, SENSOR_ID_POS


class SegmentedSensorLog(object):
    def __init__(self, sensor_log, top_compat_matrix, threshold):
        """
        Build segmented version of the given log considering the given probabilistic topological compatibility matrix.
        
        :type sensor_log: file
        :type top_compat_matrix: TopologicalCompatMatrix
        :type threshold: float
        :param sensor_log: the tab-separated file containing the sensor log.
        :param top_compat_matrix: the topological compatibility matrix of the sensor log.
        :param threshold: the threshold to reach for a direct succession to be significant.
        """
        self.sensor_log = csv.reader(sensor_log, delimiter=LOG_ENTRY_DELIMITER)
        self.top_compat_matrix = top_compat_matrix
        self.segments = []

        s0 = next(self.sensor_log, None)  # consider a sliding window of two events per step
        s1 = next(self.sensor_log, None)
        segment = [list(s0)]
        while s0 is not None and s1 is not None:
            s0_id = s0[SENSOR_ID_POS]
            s1_id = s1[SENSOR_ID_POS]

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

    def plot_stats(self):
        segments_num = len(self.segments)
        segments_lengths = [len(s) for s in self.segments]

        print('segments num:', segments_num)
        print('min length:  ', len(min(self.segments, key=len)))
        print('max length:  ', len(max(self.segments, key=len)))
        print('avg length:  ', sum(segments_lengths) / segments_num)

        plt.figure()
        sn.distplot(segments_lengths)

        plt.figure()
        sn.tsplot(segments_lengths)

        plt.show()


if __name__ == '__main__':
    SRC = os.path.join(DATA_FOLDER, 'dataset_attivita_non_innestate_filtered.tsv')
    THRESHOLD = 0.1

    with open(SRC, 'rb') as log:
        tcm = TopologicalCompatMatrix(log)

    with open(SRC, 'rb') as log:
        ssl = SegmentedSensorLog(log, tcm, THRESHOLD)
    pprint(ssl.segments)
    ssl.plot_stats()
