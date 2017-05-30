import os

from datetime import timedelta

import tensorflow as tf
import matplotlib.pyplot as plt
import time
import numpy as np
from memory_profiler import profile

from sequence_classification.sequence_classifier_input import SequenceClassifierInput
from sequence_classification.tf_rnn import RNNSequenceClassifier, NEURONS_NUM, DROPOUT_KEEP_PROB
from utils.constants import TRAINED_MODELS_FOLDER, TF_MODEL_EXT, IMG_EXT, FILENAME_SEPARATOR
from utils.files import unique_filename

CONSIDERED_LABELS = ['...']
INPUTS_PER_LABEL = 1000
EPOCHS_NUM = 10
STEPS_NUM = 100
MINI_BATCH_SIZE = 0.3


@profile
def main(considered_labels=None, cached_dataset=None, inputs_per_label=1000, ngrams_length=3):
    # retrieve input data from database
    clf_input = SequenceClassifierInput(
        considered_labels=considered_labels,
        cached_dataset=cached_dataset,
        inputs_per_label=inputs_per_label,
        ngrams_length=ngrams_length
    )

    train_data, test_data, train_labels, test_labels = clf_input.get_rnn_train_test_data()

    """
    INITIALIZE COMPUTATION GRAPH
    """
    sequence_max_length = len(train_data[0])
    frame_dimension = len(train_data[0][0])

    # sequences number (i.e. batch_size) defined at runtime
    data = tf.placeholder(tf.float32, [None, sequence_max_length, frame_dimension])
    target = tf.placeholder(tf.float32, [None, clf_input.labels_num])
    dropout_keep_prob = tf.placeholder(tf.float32)
    model = RNNSequenceClassifier(data, target, dropout_keep_prob)

    # to save and restore variables after training
    saver = tf.train.Saver()

    # start session
    start_time = time.time()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    train_size = len(train_data)
    indices_num = int(MINI_BATCH_SIZE * train_size)
    errors = []

    print('Inputs per label:  {0}'.format(clf_input.inputs_per_label))
    print('Neurons per layer: {0}'.format(NEURONS_NUM))
    print('Dropout keep prob: {0}'.format(DROPOUT_KEEP_PROB))

    for epoch in range(EPOCHS_NUM):
        print('Epoch {:2d}'.format(epoch + 1))

        for step in range(STEPS_NUM):
            print('\tstep {:3d}'.format(step + 1))
            rand_index = np.random.choice(train_size, indices_num)
            mini_batch_xs = train_data[rand_index]
            mini_batch_ys = train_labels[rand_index]
            sess.run(model.optimize, {data: mini_batch_xs, target: mini_batch_ys, dropout_keep_prob: DROPOUT_KEEP_PROB})

            # dropout_keep_prob is set to 1 (i.e. keep all) only for testing
            error = sess.run(model.error, {data: test_data, target: test_labels, dropout_keep_prob: 1})
            error_percentage = 100 * error
            errors.append(error)
            print('\taccuracy: {:3.1f}% \n\terror: {:3.1f}%'.format(100 - error_percentage, error_percentage))

    elapsed_time = (time.time() - start_time)
    print('RNN running time:', timedelta(seconds=elapsed_time))

    # save model variables
    model_checkpoint_time = str(int(time.time()))
    model_checkpoint_dir = os.path.join(TRAINED_MODELS_FOLDER, model_checkpoint_time)
    if not os.path.exists(model_checkpoint_dir):
        os.makedirs(model_checkpoint_dir)
    saver.save(sess, os.path.join(model_checkpoint_dir, model_checkpoint_time) + TF_MODEL_EXT)

    """
    PLOT ERROR FUNCTION
    """
    _, fig_basename = unique_filename(os.path.join(model_checkpoint_dir, clf_input.dump_basename))
    fig = fig_basename + IMG_EXT
    fig_zoom = FILENAME_SEPARATOR.join([fig_basename, 'zoom']) + IMG_EXT
    fig_avg = FILENAME_SEPARATOR.join([fig_basename, 'avg']) + IMG_EXT

    measures_num = EPOCHS_NUM * STEPS_NUM
    plt.figure()
    plt.plot(range(1, measures_num + 1), errors)
    plt.axis([1, measures_num, 0, 1])
    plt.savefig(fig, bbox_inches='tight')

    plt.figure()
    plt.plot(range(1, measures_num + 1), errors)
    plt.savefig(fig_zoom, bbox_inches='tight')

    plt.figure()
    # group steps errors of the same epoch and compute the average error in epoch
    plt.plot(range(1, EPOCHS_NUM + 1), [sum(group) / STEPS_NUM for group in zip(*[iter(errors)]*STEPS_NUM)])
    plt.savefig(fig_avg, bbox_inches='tight')
    # plt.show()


if __name__ == '__main__':
    # main(considered_labels=CONSIDERED_LABELS, inputs_per_label=INPUTS_PER_LABEL, ngrams_length=3)
    main(cached_dataset='1495816096_3_52483_GOOD')
