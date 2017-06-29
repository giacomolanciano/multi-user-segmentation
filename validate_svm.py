import os

import time

import pickle
from datetime import timedelta

from sklearn.externals import joblib

from sequence_classification.sequence_classifier_input import SequenceClassifierInput
from utils.dataset_management import filter_dataset
from utils.constants import TRAINED_MODELS_FOLDER, PICKLE_EXT, DATA_FOLDER, FILENAME_SEPARATOR

if __name__ == '__main__':

    NOISE_THRESHOLD = 15

    print('Loading model dump...')
    predictions_filename = os.path.join(TRAINED_MODELS_FOLDER, 'l_min_15.pkl')
    clf = joblib.load(predictions_filename)

    print('Loading validation data...')
    clf_input = SequenceClassifierInput(cached_dataset='1498490206_3_28519_GOOD_validation')
    train_data, test_data, *_ = clf_input.get_spectrum_train_test_data()  # ignoring labels

    # SequenceClassifierInput splits the dataset in train and test by default.
    # We join them to perform validation.
    validation_data = train_data + test_data

    # Filter out short sequences from dataset.
    print('Filtering dataset...')
    filter_dataset(validation_data, NOISE_THRESHOLD, clf_input.ngrams_length)
    print('\tFiltered dataset size:', str(len(validation_data)))

    # compute predictions and show stats
    print('Computing predictions...')
    start_time = time.time()
    predictions = clf.predict(validation_data)
    elapsed_time = (time.time() - start_time)

    total_sequences_num = len(validation_data)
    good_sequences_num = sum(1 for _ in filter(lambda x: x == 1, predictions))  # count positive predictions
    print('\tTime:', timedelta(seconds=elapsed_time))
    print('\tFraction of good sequences: {:3.1f}%'.format(good_sequences_num / total_sequences_num * 100))

    # dump results
    print('Dumping predictions...')
    predictions_info = [str(int(time.time())), 'l_min', str(NOISE_THRESHOLD), 'predictions']
    predictions_filename = os.path.join(DATA_FOLDER, FILENAME_SEPARATOR.join(predictions_info) + PICKLE_EXT)
    with open(predictions_filename, 'wb') as dump:
        pickle.dump(predictions, dump)
