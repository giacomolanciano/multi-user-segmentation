import os
from datetime import timedelta

from sklearn import svm
from sklearn.externals import joblib
import numpy as np
import time

from sequence_classification.spectrum_kernel import occurrence_dict_spectrum_kernel
from sequence_classification.sequence_classifier_input import SequenceClassifierInput
from utils.constants import TRAINED_MODELS_FOLDER, PICKLE_EXT

print('Loading dataset...')
clf_input = SequenceClassifierInput(cached_dataset='1497531897_3_22832_GOOD')
train_data, test_data, train_labels, test_labels = clf_input.get_spectrum_train_test_data()
train_data = train_data + test_data
train_labels = np.asarray(train_labels + test_labels)

print('Training One-class SVM...')
clf = svm.OneClassSVM(kernel=occurrence_dict_spectrum_kernel)
start_time = time.time()
clf.fit(train_data, train_labels)
elapsed_time = (time.time() - start_time)
print('\tTime: ', timedelta(seconds=elapsed_time))

print('Creating model dump...')
model_checkpoint_time = str(int(time.time()))
model_checkpoint_filename = os.path.join(TRAINED_MODELS_FOLDER, model_checkpoint_time + PICKLE_EXT)
joblib.dump(clf, model_checkpoint_filename)
