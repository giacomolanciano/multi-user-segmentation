import os
from datetime import timedelta

from sklearn import svm
from sklearn.externals import joblib
import time

from sequence_classification.spectrum_kernel import occurrence_dict_spectrum_kernel
from sequence_classification.sequence_classifier_input import SequenceClassifierInput
from utils.constants import TRAINED_MODELS_FOLDER, PICKLE_EXT


def get_real_sequence_length(sequence, ngrams_length):
    ngrams_num = sum(1 for _ in filter(None, sequence))
    return ngrams_num + ngrams_length - 1


def filter_dataset(dataset, threshold, ngrams_length):
    dataset[:] = [sequence for sequence in dataset if get_real_sequence_length(sequence, ngrams_length) >= threshold]


if __name__ == '__main__':

    print('Loading dataset...')
    clf_input = SequenceClassifierInput(cached_dataset='1497531897_3_22832_GOOD')
    train_data, test_data, *_ = clf_input.get_spectrum_train_test_data()           # ignoring labels

    # SequenceClassifierInput splits the dataset in train and test by default.
    # Since the validation is performed separately, we join the splits.
    train_data = train_data + test_data

    # Filter out short sequences from dataset.
    filter_dataset(train_data, 10, clf_input.ngrams_length)
    print(len(train_data))

    print('Training One-class SVM...')
    clf = svm.OneClassSVM(kernel=occurrence_dict_spectrum_kernel)
    start_time = time.time()
    clf.fit(train_data)
    elapsed_time = (time.time() - start_time)
    print('\tTime: ', timedelta(seconds=elapsed_time))

    print('Creating model dump...')
    model_checkpoint_time = str(int(time.time()))
    model_checkpoint_filename = os.path.join(TRAINED_MODELS_FOLDER, model_checkpoint_time + PICKLE_EXT)
    joblib.dump(clf, model_checkpoint_filename)
