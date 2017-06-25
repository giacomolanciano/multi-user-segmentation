import os

from sklearn.externals import joblib

from utils.constants import TRAINED_MODELS_FOLDER


print('Loading model dump...')
model_checkpoint_filename = os.path.join(TRAINED_MODELS_FOLDER, '1498415761.pkl')
clf = joblib.load(model_checkpoint_filename)
