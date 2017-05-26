from datetime import timedelta

from sklearn import svm
from sklearn import model_selection
import numpy as np
import time

from sequence_classification.spectrum_kernel import precomputed_occurrence_dict_spectrum_kernel
from sequence_classification.spectrum_kernel import occurrence_dict_spectrum_kernel
from sequence_classification.model_performance_measure import ModelPerformanceMeasure
from sequence_classification.sequence_classifier_input import SequenceClassifierInput

# clf = svm.SVC(kernel='precomputed')
clf = svm.OneClassSVM(kernel='precomputed')

# build training and test splits
print('Loading dataset...')
clf_input = SequenceClassifierInput(cached_dataset='1495739448_3_704_GOOD')
train_data, test_data, train_labels, test_labels = clf_input.get_spectrum_train_test_data()
train_labels = np.asarray(train_labels)
test_labels = np.asarray(test_labels)

start_time = time.time()

# pre-compute kernel matrix
print('Pre-computing kernel matrix...')
kernel_matrix_train = np.asarray(precomputed_occurrence_dict_spectrum_kernel(train_data))

clf.fit(kernel_matrix_train, train_labels)

# cross validation
# print('Performing k-fold cross validation...')
# param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10]}
# grid = model_selection.GridSearchCV(estimator=clf, param_grid=param_grid, cv=10)
# grid.fit(kernel_matrix_train, train_labels)

elapsed_time = (time.time() - start_time)

# print stats
# print('\nBest params:   %s' % grid.best_params_)
# print('Best accuracy: {:3.2f}%'.format(100 * grid.best_score_))
# print('Time: ', timedelta(seconds=elapsed_time))

# show confusion matrix for best params
# print('\nComputing confusion matrix for best params...')
# clf = svm.SVC(kernel=occurrence_dict_spectrum_kernel, C=grid.best_params_['C'])
# clf.fit(train_data, train_labels)
# pred_labels = clf.predict(test_data)
# performance_measure = ModelPerformanceMeasure(test_labels, pred_labels, CONSIDERED_CLASSES)
# performance_measure.build_confusion_matrix()
# performance_measure.plot_confusion_matrix()