import os

import unicodecsv as csv
from pprint import pprint
from topological_compat_matrix import TopologicalCompatMatrix


class SegmentedSensorLog(object):
    def __init__(self, sensor_log, top_compat_matrix, threshold, log_entry_delimiter='\t', sensor_id_pos=2):
        """
        Build segmented version of the given log considering the given probabilistic topological compatibility matrix.
        
        :type sensor_log: file
        :type top_compat_matrix: TopologicalCompatMatrix
        :type threshold: float
        :param sensor_log: the file containing the sensor log.
        :param top_compat_matrix: the topological compatibility matrix of the sensor log.
        :param threshold: the threshold to reach for a direct succession to be significant. 
        :param log_entry_delimiter: the log entry fields delimiter.
        :param sensor_id_pos: the position of the sensor id in a log entry.
        """
        self.sensor_log = csv.reader(sensor_log, delimiter=log_entry_delimiter)
        self.top_compat_matrix = top_compat_matrix
        self.segments = []

        s0 = next(self.sensor_log, None)  # consider a sliding window of two events per step
        s1 = next(self.sensor_log, None)
        segment = [s0]
        while s0 is not None and s1 is not None:
            s0_id = s0[sensor_id_pos]
            s1_id = s1[sensor_id_pos]

            if self.top_compat_matrix.prob_matrix[s0_id][s1_id] >= threshold:
                segment.append(s1)  # if above threshold, continue segment
            elif segment:
                # otherwise (if segment not empty), store a copy of the segment and restart
                self.segments.append(list(segment))
                segment = []

            # prepare next step
            s0 = s1
            s1 = next(self.sensor_log, None)


if __name__ == '__main__':
    SRC = os.path.join('data', 'dataset_attivita_non_innestate.txt')
    THRESHOLD = 0.3

    with open(SRC, 'rb') as log:
        tcm = TopologicalCompatMatrix(log)

    with open(SRC, 'rb') as log:
        ssl = SegmentedSensorLog(log, tcm, THRESHOLD)
    pprint(ssl.segments)

    tcm.plot()
