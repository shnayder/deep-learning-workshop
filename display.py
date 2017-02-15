"""
Display utilities.
"""
# (Note: I ran
# pip install pydot==1.1.0
# and pip install graphviz
# and installed the graphviz OS X package with binary libraries
# to get this to work. May not actually need the pydot downgrade...)
from IPython.display import SVG
from keras.utils.visualize_util import model_to_dot

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_keras_model(model):
    return SVG(model_to_dot(model).create(prog='dot', format='svg'))


def plot_training_curves(history):
    """
    Plot accuracy and loss curves for training and validation sets.

    Args:
        history: a Keras History.history dictionary

    Returns:
        mpl figure.
    """
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(8,2))
    ax_acc.plot(history['acc'], label='acc')
    ax_acc.plot(history['val_acc'], label='Val acc')
    ax_acc.set_xlabel('epoch')
    ax_acc.set_ylabel('accuracy')
    ax_acc.legend(loc='upper left')
    ax_acc.set_title('Accuracy')

    ax_loss.plot(history['loss'], label='loss')
    ax_loss.plot(history['val_loss'], label='Val loss')
    ax_loss.set_xlabel('epoch')
    ax_loss.set_ylabel('loss')
    ax_loss.legend(loc='upper right')
    ax_loss.set_title('Loss')

    sns.despine(fig)
    return fig
