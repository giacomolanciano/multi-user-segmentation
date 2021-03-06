import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import unicodecsv as csv

from utils.constants import LOG_ENTRY_DELIMITER, SENSOR_ID_POS


class TopologicalCompatMatrix(object):
    THRESHOLD_ERROR = 'The threshold must be a value between 0 and 1 (included).'
    Y_LABELS_ROT = 0
    X_LABELS_ROT = 90

    def __init__(self, sensor_log, sensor_id_pos=SENSOR_ID_POS):
        """
        Topological compatibility matrix associated with the given sensor log. An entry of this matrix states the
        probability that the sensor on the row is followed by the sensor on the column in the log.
        
        :type sensor_log: file
        :param sensor_log: the tab-separated file containing the sensor log.
        :param sensor_id_pos: the position of the sensor id in the log entry.
        """
        self.prob_matrix = {}
        self.sensors_occurrences = {}
        self._build_tcm(sensor_log, sensor_id_pos)

    def plot(self, threshold=None, show_values=False):
        """
        Plot the topological compatibility matrix, either probabilistic or deterministic version (if threshold is set).
        Optionally, cells' values can be shown.

        :type threshold: float
        :type show_values: bool
        :param threshold: a value between 0 and 1.
        :param show_values: a flag stating whether the cells' values must be shown or not. 
        """
        if threshold:
            if threshold < 0 or threshold > 1:
                raise ValueError(self.THRESHOLD_ERROR)
            df = pd.DataFrame(self._get_deterministic_matrix(threshold))
        else:
            df = pd.DataFrame(self.prob_matrix)
        df = df.T  # transpose the DataFrame to consider the keys of prob_matrix dict as rows

        fig = plt.figure()
        if show_values:
            plt.get_current_fig_manager().window.showMaximized()

        sn.heatmap(df, vmin=0.0, vmax=1.0, annot=show_values, square=(not show_values), cmap='Reds', linewidths=1)
        plt.ylabel('predecessor')
        plt.yticks(rotation=self.Y_LABELS_ROT)
        plt.xlabel('successor')
        plt.xticks(rotation=self.X_LABELS_ROT)
        fig.tight_layout()
        plt.show()

    """ UTILITY FUNCTIONS """

    def _build_tcm(self, sensor_log, sensor_id_pos):
        """
        Build the topological compatibility matrix associated with the given sensor log.
        
        :type sensor_log: file
        :param sensor_log: the tab-separated file containing the sensor log.
        :param sensor_id_pos: the position of the sensor id in the log entry.
        """
        sensor_log_reader = csv.reader(sensor_log, delimiter=LOG_ENTRY_DELIMITER)

        s0 = next(sensor_log_reader, None)  # consider a sliding window of two events per step
        s1 = next(sensor_log_reader, None)
        self.sensors_occurrences[s0[sensor_id_pos]] = 1
        while s0 is not None and s1 is not None:
            s0_id = s0[sensor_id_pos]
            s1_id = s1[sensor_id_pos]

            # increase sensor occurrences
            try:
                self.sensors_occurrences[s1_id] += 1
            except KeyError:
                self.sensors_occurrences[s1_id] = 1

            # add sensors ids to matrix and update succession counter
            self._add_sensor(s0_id)
            self._add_sensor(s1_id)
            self.prob_matrix[s0_id][s1_id] += 1

            # prepare next step (slide the window by one position)
            s0 = s1
            s1 = next(sensor_log_reader, None)

        for s_row in self.prob_matrix:
            for s_col in self.prob_matrix[s_row]:
                if self.prob_matrix[s_row][s_col] != 0:
                    # normalize cell value with respect to predecessor total occurrences
                    self.prob_matrix[s_row][s_col] /= self.sensors_occurrences[s_row]

    def _add_sensor(self, sensor):
        """
        Add to matrix a new row and a new column related to the given sensor (if not already existing).
        
        :type sensor: str
        :param sensor: the sensor identifier.
        """
        if sensor in self.prob_matrix:
            return  # the sensor is known, no need to add
        if not self.prob_matrix:
            # the matrix is empty, add the first sensor only
            self.prob_matrix[sensor] = {sensor: 0}
            return
        # add a row for the new sensor
        self.prob_matrix[sensor] = {key: 0 for key in self.prob_matrix.keys()}
        # add a col for the new sensor
        for s in self.prob_matrix.keys():
            self.prob_matrix[s][sensor] = 0

    def _get_deterministic_matrix(self, threshold):
        """
        Build a deterministic copy of the (probabilistic) topological compatibility matrix where all cells whose value 
        is greater or equal to the given threshold are set to 1 (0 otherwise).
        
        :type threshold: float
        :param threshold: a value between 0 and 1.
        :return: a dict of dicts representing the deterministic matrix.
        """
        if threshold < 0 or threshold > 1:
            raise ValueError(self.THRESHOLD_ERROR)

        det_matrix = {}
        for row in self.prob_matrix.keys():
            det_matrix[row] = {}
            for col in self.prob_matrix[row].keys():
                det_matrix[row][col] = 1 if self.prob_matrix[row][col] >= threshold else 0
        return det_matrix
