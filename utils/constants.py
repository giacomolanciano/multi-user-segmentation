""" Module containing all constants. """
import os

DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
LOG_EXT = '.tsv'
LOG_ENTRY_DELIMITER = '\t'
SENSOR_ID_POS = 2
SENSOR_STATE_POS = 3
SENSOR_STATE_ON = 'ON'
