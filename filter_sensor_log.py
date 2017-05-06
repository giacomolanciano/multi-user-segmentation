import os

import unicodecsv as csv

from constants import DATA_FOLDER, LOG_EXT, LOG_ENTRY_DELIMITER, SENSOR_STATE_POS, SENSOR_STATE_ON


def filter_sensor_log(sensor_log):
    """
    Remove from the given sensor log all the entries related to a reset of a sensor and save the result to file.

    :type sensor_log: file
    :param sensor_log: the tab-separated file containing the sensor log.
    """
    dest = sensor_log.name[:-len(LOG_EXT)] + '_filtered' + LOG_EXT
    src_reader = csv.reader(sensor_log, delimiter=LOG_ENTRY_DELIMITER)

    with open(dest, 'wb') as filtered_log:
        dest_writer = csv.writer(filtered_log, delimiter=LOG_ENTRY_DELIMITER)

        entry = next(src_reader, None)
        while entry is not None:
            if entry[SENSOR_STATE_POS] == SENSOR_STATE_ON:
                dest_writer.writerow(entry)
            entry = next(src_reader, None)


if __name__ == '__main__':
    SRC = os.path.join(DATA_FOLDER, 'dataset_attivita_non_innestate.tsv')
    with open(SRC, 'rb') as log:
        filter_sensor_log(log)
