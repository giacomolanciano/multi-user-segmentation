import os

from models.topological_compat_matrix import TopologicalCompatMatrix
from utils.constants import DATA_FOLDER

if __name__ == '__main__':
    SRC = os.path.join(DATA_FOLDER, 'dataset_attivita_non_innestate_filtered_simplified.txt')
    SENSOR_ID_POS_ = 0

    with open(SRC, 'rb') as log:
        tcm = TopologicalCompatMatrix(log, sensor_id_pos=SENSOR_ID_POS_)
    tcm.plot(show_values=False)
