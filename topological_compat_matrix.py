import os

import unicodecsv as csv
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


class TopologicalCompatMatrix(object):
    PLOT_SIZE = 10
    Y_LABELS_ROT = 0
    X_LABELS_ROT = 90

    def __init__(self, sensor_log, log_entry_delimiter='\t', sensor_id_pos=2):
        """
        Build the topological compatibility matrix associated with the given sensor log.
        
        :type sensor_log: file
        :param sensor_log: the file containing the sensor log.
        :param log_entry_delimiter: the log entry fields delimiter.
        :param sensor_id_pos: the position of the sensor id in a log entry.
        """
        self.sensor_log = csv.reader(sensor_log, delimiter=log_entry_delimiter)
        self.prob_matrix = {}
        self.sensors_occurrences = {}

        s0 = next(self.sensor_log, None)  # consider a sliding window of two events per step
        s1 = next(self.sensor_log, None)
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
            self.add_sensor(s0_id)
            self.add_sensor(s1_id)
            self.prob_matrix[s0_id][s1_id] += 1

            # prepare next step
            s0 = s1
            s1 = next(self.sensor_log, None)

        for s_row in self.prob_matrix:
            for s_col in self.prob_matrix[s_row]:
                if self.prob_matrix[s_row][s_col] != 0:
                    # normalize cell value with respect to antecedent total occurrences
                    self.prob_matrix[s_row][s_col] /= self.sensors_occurrences[s_row]

    def add_sensor(self, sensor):
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

    def plot(self):
        df = pd.DataFrame(self.prob_matrix)
        plt.figure(figsize=(self.PLOT_SIZE, self.PLOT_SIZE))
        sn.heatmap(df, square=True, cmap='Blues', linewidths=1)
        plt.yticks(rotation=self.Y_LABELS_ROT)
        plt.xticks(rotation=self.X_LABELS_ROT)
        plt.show()


if __name__ == '__main__':
    SRC = os.path.join('data', 'dataset_attivita_non_innestate.txt')
    with open(SRC, 'rb') as log:
        tcm = TopologicalCompatMatrix(log)
    tcm.plot()