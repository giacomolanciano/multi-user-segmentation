import os

import unicodecsv as csv

from utils.constants import DATA_FOLDER, LOG_EXT, LOG_ENTRY_DELIMITER, SENSOR_ID_POS


def simplify_sensor_log(sensor_log, readable=True):
    """
    Translate the given sensor log in a symbols sequence that can be processed by MIM
    (http://web.tecnico.ulisboa.pt/diogo.ferreira/mimcode/).
    Notice that if readability of results is important, then MIM code allows only for a maximum number of distinct 
    symbols equals to the size of English alphabet. Hence, in that case, a mapping between sensor ids and letters is 
    automatically computed.

    :type sensor_log: file
    :param sensor_log: the tab-separated file containing the sensor log.
    :param readable: whether the mapping between sensor ids and letters has to be computed or not.
    """
    SYMBOLS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
               'V', 'W', 'X', 'Y', 'Z']

    dest = sensor_log.name[:-len(LOG_EXT)] + '_simplified.txt'
    dest_dict = sensor_log.name[:-len(LOG_EXT)] + '_simplified_dict.txt'
    src_reader = csv.reader(sensor_log, delimiter=LOG_ENTRY_DELIMITER)
    sensor_id_dict = {}

    with open(dest, 'w') as simplified_log:
        entry = next(src_reader, None)
        while entry is not None:
            sensor_id = entry[SENSOR_ID_POS]

            if readable:
                try:
                    translation = sensor_id_dict[sensor_id]
                except KeyError:
                    translation = SYMBOLS[len(sensor_id_dict)]
                    sensor_id_dict[sensor_id] = translation
            else:
                translation = sensor_id

            simplified_log.write(translation + '\n')
            entry = next(src_reader, None)

    with open(dest_dict, 'w') as simplified_log_dict:
        for k, v in sensor_id_dict.items():
            simplified_log_dict.write('%s \t\t %s\n' % (v, k))


if __name__ == '__main__':
    SRC = os.path.join(DATA_FOLDER, 'dataset_attivita_non_innestate_filtered.tsv')
    with open(SRC, 'rb') as log:
        simplify_sensor_log(log)
