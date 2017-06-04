import os

from models.topological_compat_matrix import TopologicalCompatMatrix
from utils.constants import DATA_FOLDER

if __name__ == '__main__':
    SRC = os.path.join(DATA_FOLDER, 'dataset_attivita_non_innestate_filtered.tsv')
    with open(SRC, 'rb') as log:
        tcm = TopologicalCompatMatrix(log)
    tcm.plot(show_values=False)
