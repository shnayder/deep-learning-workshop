from __future__ import print_function
import numpy as np
from six.moves import cPickle as pickle
from six.moves import range
import os

def load_data(path='notMNIST.pickle'):
    """
    Read a pre-prepared pickled dataset of the notmnist dataset
    (http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html).

    Args:
        path: Where to find the file, relative to .

    Returns: tuple of three pairs of np arrays
        (train_x, train_y), (valid_x, valid_y), (test_x, test_y)
        The x arrays have shape (-1,28,28), and x values are floating point, normalized to have approximately zero mean
        and standard deviation ~0.5.

        The y arrays are single dimensional categorical (not 1-hot).
    """
    with open(os.path.expanduser(path), 'rb') as f:
      save = pickle.load(f)
      train_x = save['train_dataset']
      train_y = save['train_labels']
      valid_x = save['valid_dataset']
      valid_y = save['valid_labels']
      test_x = save['test_dataset']
      test_y = save['test_labels']
      del save  # hint to help gc free up memory

      return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)
