import matplotlib.pyplot as plt
import seaborn as sn
import unicodecsv as csv

from models.topological_compat_matrix import TopologicalCompatMatrix
from utils.constants import LOG_ENTRY_DELIMITER, SENSOR_ID_POS, NOISE_THRESHOLD


class BStep(object):
    def __init__(self):
        """
        Gather the two sets of segments that must be combined to contribute to the validation set.
        """
        self.closed_segments = []  # the segments closed at this B-step.
        self.compat_segments = []  # the segments opened between this B-step and the next that are compatible.

    def add_closed_segment(self, s):
        self.closed_segments.append(s)

    def add_compat_segment(self, s):
        self.compat_segments.append(s)


class SegmentedSensorLog(object):
    def __init__(self, sensor_log=None, top_compat_matrix=None, compat_threshold=None, segments=None,
                 sensor_id_pos=SENSOR_ID_POS, noise_threshold=NOISE_THRESHOLD):
        """
        Segmented version of the given log, built according to the given probabilistic topological compatibility matrix.
        
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
            self.compat_threshold = compat_threshold
            self.noise_threshold = noise_threshold
            self.sensor_id_pos = sensor_id_pos

            self.b_steps = []  # the B-steps performed during the segmentation (to be used in validation).

            self._find_segments(sensor_log)

        else:
            raise ValueError('Not enough inputs provided.')

    def plot_stats(self, distribution=True, time_series=False):
        """
        Visualize segmented sensor log statistics.
        Notice that time-series visualization is significant only when the segments are chronologically ordered.
        
        :param distribution: whether the distribution visualization must be shown.
        :param time_series: whether the time-series visualization must be shown.
        """
        segments_num = len(self.segments)
        segments_lengths = [len(s) for s in self.segments]

        print('segments num:', segments_num)
        print('min length:  ', len(min(self.segments, key=len)))
        print('max length:  ', len(max(self.segments, key=len)))
        print('avg length:  ', int(sum(segments_lengths) / segments_num))
        print('b-steps num: ', len(self.b_steps))

        if distribution:
            plt.figure()
            sn.distplot(segments_lengths)

        if time_series:
            plt.figure()
            sn.tsplot(segments_lengths)

        plt.show()

    """ UTILITY FUNCTIONS """

    def _find_segments(self, sensor_log):
        """
        Find segments in the given sensor log.

        :type sensor_log: file
        :param sensor_log: the tab-separated file containing the sensor log.
        """
        sensor_log_reader = csv.reader(sensor_log, delimiter=LOG_ENTRY_DELIMITER)

        open_segments = []
        for measure in sensor_log_reader:
            sensor_id = measure[self.sensor_id_pos]

            # find compatible open segments
            compat_segments_idxs = self._get_compat_segments_indices(open_segments, sensor_id)

            # check compatibility results
            if len(compat_segments_idxs) == 1:
                # only one compat segment exists, append the measure
                segment_idx = compat_segments_idxs[0]
                open_segments[segment_idx].append(measure)

            else:
                if len(compat_segments_idxs) > 1:
                    # if many compat segments exist, close them (B-step)
                    self._close_segments(open_segments, compat_segments_idxs)

                # open new segment and append the measure
                new_segment = [measure]
                open_segments.append(new_segment)

                # check whether the new segment is compatible with at least a segment in last B-step
                if self.b_steps:
                    if self._get_compat_segments_indices(self.b_steps[-1].closed_segments, sensor_id):
                        # add new segment to last B-step compatibility list
                        self.b_steps[-1].add_compat_segment(new_segment)

        # close remaining open segments
        self._close_segments(open_segments)

    def _get_compat_segments_indices(self, segments, sensor_id):
        """
        Return tho positions of the segments that are compatible (according to the given threshold) with the provided
        sensor identifier.

        :type segments: list
        :param segments: the list of open segments.
        :param sensor_id: the sensor identifier.
        :return: a list containing the indices of the compatible segments.
        """
        compat_segments_idxs = []
        for idx, os in enumerate(segments):
            # consider the sensor id of the last measure in segment
            old_sensor_id = os[-1][self.sensor_id_pos]

            if self.top_compat_matrix.prob_matrix[old_sensor_id][sensor_id] >= self.compat_threshold:
                # the direct succession value is above the threshold -> segment is compatible
                compat_segments_idxs.append(idx)
        return compat_segments_idxs

    def _close_segments(self, open_segments, indices=None):
        """
        Remove the segments identified by the given indices from the open list and add them to close list (if the
        minimum length is matched). If the indices are not provided, then all segments will be closed.

        :type open_segments: list
        :type indices: list
        :param open_segments: the list of open segments.
        :param indices: the positions of the segments to be closed.
        """
        if indices:
            indices = sorted(indices)
        else:
            # build a list of all possible indices to use the same approach for both cases.
            indices = [x for x in range(len(open_segments))]

        # add all segments to be closed to the global segments list and group them in a B-step
        b_step = BStep()
        for idx in reversed(indices):
            closed_segment = open_segments.pop(idx)          # remove segment from open list
            if len(closed_segment) >= self.noise_threshold:  # noise filtering
                self.segments.append(closed_segment)
                b_step.add_closed_segment(closed_segment)
        if b_step.closed_segments:
            self.b_steps.append(b_step)

    """ DEPRECATED FUNCTIONS """

    def _find_segments_old(self, sensor_log):
        """
        Find segments in the given sensor log (old version).

        :type sensor_log: file
        :param sensor_log: the tab-separated file containing the sensor log.
        """
        sensor_log_reader = csv.reader(sensor_log, delimiter=LOG_ENTRY_DELIMITER)

        s0 = next(sensor_log_reader, None)  # consider a sliding window of two events per step
        s1 = next(sensor_log_reader, None)
        segment = [list(s0)]
        while s0 is not None and s1 is not None:
            s0_id = s0[self.sensor_id_pos]
            s1_id = s1[self.sensor_id_pos]

            if self.top_compat_matrix.prob_matrix[s0_id][s1_id] >= self.compat_threshold:
                # the direct succession value is above the threshold
                segment.append(list(s1))  # continue the segment
            else:
                # the direct succession value is under the threshold
                if len(segment) >= self.noise_threshold:  # only segments longer than a threshold are considered
                    self.segments.append(list(segment))   # store a copy of the segment so far
                segment = [list(s1)]                      # start the new segment from the second item in the window

            # prepare next step (slide the window by one position)
            s0 = s1
            s1 = next(sensor_log_reader, None)
