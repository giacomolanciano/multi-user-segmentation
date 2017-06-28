import os
from datetime import timedelta

from sklearn import svm
from sklearn.externals import joblib
import time

from sequence_classification.spectrum_kernel import occurrence_dict_spectrum_kernel
from sequence_classification.sequence_classifier_input import SequenceClassifierInput
from utils.constants import TRAINED_MODELS_FOLDER, PICKLE_EXT
from utils.dataset_management import filter_dataset


if __name__ == '__main__':

    NOISE_THRESHOLD = 10

    print('Loading dataset...')
    clf_input = SequenceClassifierInput(cached_dataset='1498483802_3_17732_GOOD_training')
    train_data, test_data, *_ = clf_input.get_spectrum_train_test_data()  # ignoring labels

    # SequenceClassifierInput splits the dataset in train and test by default.
    # Since the validation is performed separately, we join the splits.
    train_data = train_data + test_data

    # Filter out short sequences from dataset.
    print('Filtering dataset...')
    filter_dataset(train_data, NOISE_THRESHOLD, clf_input.ngrams_length)

    print('Training One-class SVM...')
    clf = svm.OneClassSVM(kernel=occurrence_dict_spectrum_kernel)
    start_time = time.time()
    clf.fit(train_data)
    elapsed_time = (time.time() - start_time)
    print('\tTime:', timedelta(seconds=elapsed_time))
    print('\tNoise threshold:', NOISE_THRESHOLD)

    print('Creating model dump...')
    model_checkpoint_time = str(int(time.time()))
    model_checkpoint_filename = os.path.join(TRAINED_MODELS_FOLDER, model_checkpoint_time + PICKLE_EXT)
    joblib.dump(clf, model_checkpoint_filename)
    print(model_checkpoint_filename)
