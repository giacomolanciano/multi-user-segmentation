import os

from models.segmented_sensor_log import SegmentedSensorLog
from models.topological_compat_matrix import TopologicalCompatMatrix
from sequence_classification.sequence_classifier_input import SequenceClassifierInput
from utils.constants import DATA_FOLDER


if __name__ == '__main__':
    # import pickle
    # from pprint import pprint

    # SRC = os.path.join(DATA_FOLDER, 'sequences.pkl')
    #
    # with open(SRC, 'rb') as src:
    #     segs = dict(pickle.load(src))
    #
    # ssl = SegmentedSensorLog(segments=segs.keys())

    SRC = os.path.join(DATA_FOLDER, 'complete_dataset_preprocessed_filtered_simplified.txt')
    THRESHOLD = 0.1

    with open(SRC, 'rb') as log:
        tcm = TopologicalCompatMatrix(log, sensor_id_pos=0)

    with open(SRC, 'rb') as log:
        ssl = SegmentedSensorLog(log, tcm, THRESHOLD, sensor_id_pos=0)

    # GOOD = 'GOOD'
    # sequences = []
    # labels = []
    # for segment_ in ssl.segments:
    #     if len(segment_) > 1:
    #         sequence = ''
    #         for c in segment_:
    #             sequence += c[0]
    #         sequences.append(sequence)
    #         labels.append(GOOD)
    #
    # clf_input = SequenceClassifierInput(sequences=sequences, labels=labels)
    # clf_input.get_spectrum_train_test_data()
