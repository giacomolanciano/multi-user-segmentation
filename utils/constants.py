""" Module containing all constants. """
import os

DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
DATABASE = os.path.join(DATA_FOLDER, '')  # add .db file in data/ if needed
LOG_EXT = '.tsv'
LOG_ENTRY_DELIMITER = '\t'
SENSOR_ID_POS = 2
SENSOR_STATE_POS = 3
SENSOR_STATE_ON = 'ON'
NOISE_THRESHOLD = 2

TRAINED_MODELS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'trained_classifiers')
TF_MODEL_EXT = '.ckpt'
IMG_EXT = '.png'
PADDING_VALUE = 0

FILENAME_SEPARATOR = '_'
RNN_SUFFIX = 'rnn'
SPECTRUM_SUFFIX = 'spectrum'
GLOVE_TRAIN_SUFFIX = 'glove_matrix_train.mmap'
GLOVE_TEST_SUFFIX = 'glove_matrix_test.mmap'

SPECTRUM_KEY = 'spectrum'
LABELS_KEY = 'labels'
INPUTS_PER_LABEL_KEY = 'ipl'
TIME_KEY = 'time'
TRAIN_DATA_KEY = 'train_data'
TEST_DATA_KEY = 'test_data'
TRAIN_LABELS_KEY = 'train_labels'
TEST_LABELS_KEY = 'test_labels'
GLOVE_EMBEDDING_SIZE_KEY = 'glove_embedding_size'
MAX_COLS_NUM_KEY = 'max_cols_num'

TRAIN_DATA_POS = 0
TEST_DATA_POS = 1
TRAIN_LABELS_POS = 2
TEST_LABELS_POS = 3

PICKLE_EXT = '.pkl'
