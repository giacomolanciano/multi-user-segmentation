def filter_dataset(dataset, threshold, ngrams_length):
    """
    Filter out of the dataset all the sequences that do not match the minimum length (expressed by threshold).

    :param dataset: the dataset containing sequences n-grams representations.
    :param threshold: the minimum length to be matched.
    :param ngrams_length: the length of each n-gram.
    """
    dataset[:] = [sequence for sequence in dataset if _get_actual_sequence_length(sequence, ngrams_length) >= threshold]


def _get_actual_sequence_length(sequence, ngrams_length):
    """
    Compute the length of a sequence given its n-grams representation and n-grams length.

    :param sequence: the n-grams representation of the sequence.
    :param ngrams_length: the length of each n-gram.
    :return: the length of the sequence.
    """
    ngrams_num = sum(1 for _ in filter(None, sequence))
    return ngrams_num + ngrams_length - 1
