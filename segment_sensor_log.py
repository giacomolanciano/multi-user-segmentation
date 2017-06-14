import os

from models.segmented_sensor_log import SegmentedSensorLog
from models.topological_compat_matrix import TopologicalCompatMatrix
from sequence_classification.sequence_classifier_input import SequenceClassifierInput
from utils.constants import DATA_FOLDER


def _build_sequence_clf_training_set():
    GOOD = 'GOOD'
    sequences = []
    labels = []
    for segment_ in ssl.segments:
        sequence = ''
        for c in segment_:
            sequence += c[0]
        sequences.append(sequence)
        labels.append(GOOD)
    clf_input = SequenceClassifierInput(sequences=sequences, labels=labels)
    clf_input.get_spectrum_train_test_data()


if __name__ == '__main__':
    SRC = os.path.join(DATA_FOLDER, 'dataset_attivita_non_innestate_filtered_simplified.txt')
    COMPAT_THRESHOLD_ = 0.1
    NOISE_THRESHOLD_ = 2
    SENSOR_ID_POS_ = 0

    with open(SRC, 'rb') as log:
        tcm = TopologicalCompatMatrix(sensor_log=log, sensor_id_pos=SENSOR_ID_POS_)

    with open(SRC, 'rb') as log:
        ssl = SegmentedSensorLog(sensor_log=log, top_compat_matrix=tcm, sensor_id_pos=SENSOR_ID_POS_,
                                 compat_threshold=COMPAT_THRESHOLD_, noise_threshold=NOISE_THRESHOLD_)

    # show the segments
    # from pprint import pprint
    # pprint(ssl.segments)

    # build a training set for a sequence classifier
    # _build_sequence_clf_training_set()

    # show segmented log statistics
    ssl.plot_stats()
