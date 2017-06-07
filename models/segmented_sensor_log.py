import matplotlib.pyplot as plt
import seaborn as sn
import unicodecsv as csv

from models.topological_compat_matrix import TopologicalCompatMatrix
from utils.constants import LOG_ENTRY_DELIMITER, SENSOR_ID_POS


class SegmentedSensorLog(object):
    def __init__(self, sensor_log=None, top_compat_matrix=None, compat_threshold=None, segments=None,
                 sensor_id_pos=SENSOR_ID_POS, noise_threshold=2):
        """
        Build segmented version of the given log considering the given probabilistic topological compatibility matrix.
        
        :type sensor_log: file
        :type top_compat_matrix: TopologicalCompatMatrix
        :type compat_threshold: float
        :type segments: list
        :type sensor_id_pos: int
        :type noise_threshold: int
        :param sensor_log: the tab-separated file containing the sensor log.
        :param top_compat_matrix: the topological compatibility matrix of the sensor log.
        :param compat_threshold: the threshold to reach for a direct succession to be significant.
        :param segments: a precomputed list of segments.
        :param sensor_id_pos: the position of the sensor id in the log entry.
        :param noise_threshold: the minimum length of a segment.
        """
        if segments:
            self.segments = segments

        elif sensor_log and top_compat_matrix and compat_threshold:
            self.segments = []
            self.top_compat_matrix = top_compat_matrix
            self._find_segments(sensor_log, compat_threshold, sensor_id_pos, noise_threshold)

        else:
            raise ValueError('Not enough inputs provided.')

    def plot_stats(self, distribution=True, time_series=True):
        """
        Visualize segmented sensor log statistics.
        Notice that time-series visualization is significant only when the segments are chronologically ordered.
        
        :param distribution: whether the distribution visualization must be shown.
        :param time_series: whether the time-series visualization must be shown.
        """
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

    """ UTILITY FUNCTIONS """

    def _find_segments(self, sensor_log, threshold, sensor_id_pos, noise_threshold):
        """
        Find segments in the given sensor log.

        :type sensor_log: file
        :type threshold: float
        :type sensor_id_pos: int
        :type noise_threshold: int
        :param sensor_log: the tab-separated file containing the sensor log.
        :param threshold: the threshold to reach for a direct succession to be significant.
        :param sensor_id_pos: the position of the sensor id in the log entry.
        :param noise_threshold: the minimum length of a segment.
        """
        sensor_log_reader = csv.reader(sensor_log, delimiter=LOG_ENTRY_DELIMITER)

        s0 = next(sensor_log_reader, None)  # consider a sliding window of two events per step
        s1 = next(sensor_log_reader, None)
        segment = [list(s0)]
        while s0 is not None and s1 is not None:
            s0_id = s0[sensor_id_pos]
            s1_id = s1[sensor_id_pos]

            if self.top_compat_matrix.prob_matrix[s0_id][s1_id] >= threshold:
                # the direct succession value is above the threshold
                segment.append(list(s1))  # continue the segment
            else:
                # the direct succession value is under the threshold
                if len(segment) >= noise_threshold:      # only segments longer than a threshold are considered
                    self.segments.append(list(segment))  # store a copy of the segment so far
                segment = [list(s1)]                     # start the new segment from the second item in the window

            # prepare next step (slide the window by one position)
            s0 = s1
            s1 = next(sensor_log_reader, None)
