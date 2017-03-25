"""
Display utilities.
"""
# (Note: I ran
# pip install pydot==1.1.0
# and pip install graphviz
# and installed the graphviz OS X package with binary libraries
# to get this to work. May not actually need the pydot downgrade...)
from IPython.display import SVG

# TODO: support Keras 2 and earlier
from keras.utils.vis_utils import model_to_dot

import itertools as it
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import confusion_matrix


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
    if 'acc' in history:
        ax_acc.plot(history['acc'], label='acc')
        if 'val_acc' in history:
            ax_acc.plot(history['val_acc'], label='Val acc')
        ax_acc.set_xlabel('epoch')
        ax_acc.set_ylabel('accuracy')
        ax_acc.legend(loc='upper left')
        ax_acc.set_title('Accuracy')

    ax_loss.plot(history['loss'], label='loss')
    if 'val_loss' in history:
        ax_loss.plot(history['val_loss'], label='Val loss')
    ax_loss.set_xlabel('epoch')
    ax_loss.set_ylabel('loss')
    ax_loss.legend(loc='upper right')
    ax_loss.set_title('Loss')

    sns.despine(fig)
    return fig


def plot_confusion_matrix(labels, predictions,
                          classes,
                          normalize=False,
                          title="Confusion matrix",
                          cmap=plt.cm.Blues):
    """
    Plot a confusion matrix for predictions vs labels.
    Both should be one-hot.

    Based on.
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    # convert from one-hot
    cat_labels = np.argmax(labels, axis=1)
    cat_predicts = np.argmax(predictions, axis=1)

    cm = confusion_matrix(cat_labels, cat_predicts)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)


    #print(cm)

    thresh = cm.max() / 2.
    for i, j in it.product(range(cm.shape[0]), range(cm.shape[1])):
        if 0 < cm[i,j] < 1:
            val = "{:.2f}".format(cm[i,j])
        else:
            val = cm[i,j]
        plt.text(j, i, val,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return cm
