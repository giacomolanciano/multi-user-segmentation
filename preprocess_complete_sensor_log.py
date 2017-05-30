import os

import unicodecsv as csv

from utils.constants import DATA_FOLDER, LOG_EXT, LOG_ENTRY_DELIMITER


def preprocess_complete_sensor_log(sensor_log):
    """
    Convert the complete sensor log to the standard form used in the project.

    :type sensor_log: file
    :param sensor_log: the tab-separated file containing the sensor log.
    """
    date_time_pos = 0
    sensor_id_pos = 2
    sensor_state_pos = 3
    time_zone_length = 3

    dest = os.path.splitext(sensor_log.name)[0] + '_preprocessed' + LOG_EXT
    src_reader = csv.reader(sensor_log, delimiter='|')

    with open(dest, 'wb') as preprocessed_log:
        dest_writer = csv.writer(preprocessed_log, delimiter=LOG_ENTRY_DELIMITER)

        entry = next(src_reader, None)
        while entry is not None:
            date_time = str(entry[date_time_pos]).strip().split()
            date = date_time[0]
            time = date_time[1][:-time_zone_length]  # trim time zone

            sensor_id = str(entry[sensor_id_pos]).strip()
            sensor_state = str(entry[sensor_state_pos]).strip()

            dest_writer.writerow((date, time, sensor_id, sensor_state))
            entry = next(src_reader, None)


if __name__ == '__main__':
    SRC = os.path.join(DATA_FOLDER, 'complete_dataset.txt')
    with open(SRC, 'rb') as log:
        preprocess_complete_sensor_log(log)
