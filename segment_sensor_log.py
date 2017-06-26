import os

from models.segmented_sensor_log import SegmentedSensorLog
from models.topological_compat_matrix import TopologicalCompatMatrix
from sequence_classification.sequence_classifier_input import SequenceClassifierInput
from utils.constants import DATA_FOLDER

GOOD_LABEL = 'GOOD'


def build_sequence_clf_training_set(segmented_log, sensor_id_pos, ngrams_length):
    print('Building training set...')
    sequences = []
    labels = []
    for segment_ in segmented_log.segments:
        sequence = ''
        for c in segment_:
            sequence += c[sensor_id_pos]
        sequences.append(sequence)
        labels.append(GOOD_LABEL)
    clf_input = SequenceClassifierInput(sequences=sequences, labels=labels, ngrams_length=ngrams_length)
    train_data, *_ = clf_input.get_spectrum_train_test_data()
    return len(train_data[0])  # return the max sequence length


def build_sequence_clf_validation_set(segmented_log, sensor_id_pos, ngrams_length, max_vector_length):
    print('Building validation set...')
    sequences = []
    labels = []
    for b_step in segmented_log.b_steps:
        # compute the cartesian product of the two collections of segments in the current b step.
        for segment_ in b_step.segments:
            for compat_segment_ in b_step.compat_segments:
                sequence = ''
                for c in segment_ + compat_segment_:
                    sequence += c[sensor_id_pos]
                sequences.append(sequence)
                labels.append(GOOD_LABEL)
    clf_input = SequenceClassifierInput(sequences=sequences, labels=labels, ngrams_length=ngrams_length)
    clf_input.get_spectrum_train_test_data(max_vector_length)


if __name__ == '__main__':
    import time
    from datetime import timedelta

    SRC = os.path.join(DATA_FOLDER, 'complete_dataset_preprocessed_filtered_simplified.txt')
    COMPAT_THRESHOLD_ = 0.1
    NOISE_THRESHOLD_ = 3
    SENSOR_ID_POS_ = 0
    NGRAMS_LENGTH = 3

    if NOISE_THRESHOLD_ < NGRAMS_LENGTH:
        raise ValueError('The minimum sequence length must be greater or equal than n-grams length.')

    start_time = time.time()

    print('Building topological compatibility matrix...')
    with open(SRC, 'rb') as log:
        tcm = TopologicalCompatMatrix(sensor_log=log, sensor_id_pos=SENSOR_ID_POS_)

    print('Performing log segmentation...')
    with open(SRC, 'rb') as log:
        ssl = SegmentedSensorLog(sensor_log=log, top_compat_matrix=tcm, sensor_id_pos=SENSOR_ID_POS_,
                                 compat_threshold=COMPAT_THRESHOLD_, noise_threshold=NOISE_THRESHOLD_)

    elapsed_time = (time.time() - start_time)
    print('Segmentation time:', timedelta(seconds=elapsed_time))

    # show segments
    # from pprint import pprint
    # pprint(ssl.segments)

    # show b-steps
    # for b in ssl.b_steps:
    #     pprint(b.segments)
    #     pprint(b.compat_segments)
    #     print()

    # build a training set for a sequence classifier
    # max_vector_length_ = build_sequence_clf_training_set(ssl, SENSOR_ID_POS_, NGRAMS_LENGTH)

    # build a validation set for a sequence classifier
    # build_sequence_clf_validation_set(ssl, SENSOR_ID_POS_, NGRAMS_LENGTH, max_vector_length_)

    # show segmented log statistics
    ssl.plot_stats()
