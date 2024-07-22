import numpy as np

def pad_sequences(sequences, maxlen):
    """
    Pads a list of sequences to the same length with NaNs.

    Args:
        sequences (list of lists): Sequences to be padded.
        maxlen (int): Length to pad each sequence to.

    Returns:
        np.ndarray: 2D array with padded sequences.
    """
    padded_sequences = np.full((len(sequences), maxlen), np.nan)
    for i, seq in enumerate(sequences):
        length = len(seq)
        padded_sequences[i, :length] = seq
    return padded_sequences