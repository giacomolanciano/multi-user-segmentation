import os

from sklearn.externals import joblib

from sequence_classification.sequence_classifier_input import SequenceClassifierInput
from train_svm import filter_dataset
from utils.constants import TRAINED_MODELS_FOLDER


if __name__ == '__main__':

    NOISE_THRESHOLD = 10

    print('Loading model dump...')
    model_checkpoint_filename = os.path.join(TRAINED_MODELS_FOLDER, '...')
    clf = joblib.load(model_checkpoint_filename)

    clf_input = SequenceClassifierInput(cached_dataset='1498490206_3_28519_GOOD_validation')
    train_data, test_data, *_ = clf_input.get_spectrum_train_test_data()  # ignoring labels

    # SequenceClassifierInput splits the dataset in train and test by default.
    # We join them to perform validation.
    validation_data = train_data + test_data

    # Filter out short sequences from dataset.
    print('Filtering dataset...')
    filter_dataset(validation_data, NOISE_THRESHOLD, clf_input.ngrams_length)

    # compute predictions and show stats
    predictions = clf.predict(validation_data)
    total_sequences_num = len(validation_data)
    good_sequences_num = sum(1 for _ in filter(lambda x: x == 1, predictions))
    print('Fraction of good sequences: {:3.1f}%'.format(good_sequences_num / total_sequences_num))
