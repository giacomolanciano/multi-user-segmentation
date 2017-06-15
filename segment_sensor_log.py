import os

from models.segmented_sensor_log import SegmentedSensorLog
from models.topological_compat_matrix import TopologicalCompatMatrix
from sequence_classification.sequence_classifier_input import SequenceClassifierInput
from utils.constants import DATA_FOLDER


def _build_sequence_clf_training_set(segmented_log):
    GOOD = 'GOOD'
    sequences = []
    labels = []
    for segment_ in segmented_log.segments:
        sequence = ''
        for c in segment_:
            sequence += c[0]
        sequences.append(sequence)
        labels.append(GOOD)
    clf_input = SequenceClassifierInput(sequences=sequences, labels=labels)
    clf_input.get_spectrum_train_test_data()


if __name__ == '__main__':
    import time
    from datetime import timedelta

    SRC = os.path.join(DATA_FOLDER, 'complete_dataset_preprocessed_filtered_simplified.txt')
    COMPAT_THRESHOLD_ = 0.1
    NOISE_THRESHOLD_ = 2
    SENSOR_ID_POS_ = 0

    start_time = time.time()

    with open(SRC, 'rb') as log:
        tcm = TopologicalCompatMatrix(sensor_log=log, sensor_id_pos=SENSOR_ID_POS_)

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
    # _build_sequence_clf_training_set(ssl)

    # show segmented log statistics
    ssl.plot_stats()
