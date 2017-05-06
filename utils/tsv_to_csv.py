import os

import unicodecsv as csv

from utils.constants import DATA_FOLDER, LOG_EXT, LOG_ENTRY_DELIMITER


def convert_to_csv(sensor_log):
    """
    Convert the sensor log (tab-separated) to a comma-separated file.
    
    :type sensor_log: file
    :param sensor_log: the tab-separated file containing the sensor log.
    """
    dest = sensor_log.name[:-len(LOG_EXT)] + '.csv'
    src_reader = csv.reader(sensor_log, delimiter=LOG_ENTRY_DELIMITER)

    with open(dest, 'wb') as csv_log:
        dest_writer = csv.writer(csv_log, delimiter=',')

        entry = next(src_reader, None)
        while entry is not None:
            dest_writer.writerow(entry[:3])  # retain only: day, timestamp, sensor_id
            entry = next(src_reader, None)


if __name__ == '__main__':
    SRC = os.path.join(DATA_FOLDER, 'dataset_attivita_non_innestate_filtered.tsv')
    with open(SRC, 'rb') as log:
        convert_to_csv(log)
