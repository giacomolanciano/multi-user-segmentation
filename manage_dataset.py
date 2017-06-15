import os

import klepto

from utils.constants import DATA_FOLDER


def dump_dataset(dataset_dict, filename):
    """
    Create a dump (named as 'filename') of the given dataset in secondary storage.
    
    :type dataset_dict: dict
    :type filename: str
    :param dataset_dict: a dict representing the feature of the dataset.
    :param filename: the filename of the dataset dump.
    """
    dirname = os.path.join(DATA_FOLDER, filename)
    archive = klepto.archives.dir_archive(dirname, cached=True, serialized=True)
    for key, val in dataset_dict.items():
        archive[key] = val
    archive.dump()


def load_dataset(cached_dataset):
    """
    Load a dataset in memory from a dump in secondary storage identified by the given filename.
    
    :type cached_dataset: str
    :param cached_dataset: the filename of the dataset.
    :return: the object that represents the dataset.
    """
    dirname = os.path.join(DATA_FOLDER, cached_dataset)
    archive = klepto.archives.dir_archive(dirname, cached=True, serialized=True)
    archive.load()
    return archive

if __name__ == '__main__':
    from pprint import pprint

    d = load_dataset(cached_dataset='1497531897_3_22832_GOOD_spectrum')
    pprint(d['train_data'][:10])

    # insert the code for the desired manipulation of a cached dataset.
