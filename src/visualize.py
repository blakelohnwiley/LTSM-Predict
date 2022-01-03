# import packages
import io
from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from static import window_len, target_col

"""
Author: Blake Lohn-Wiley (maths.lohnwiley@gmail.com)
Date: 03.01.2022
"""

def line_plot(line1, line2, label1=None, label2=None, title='', lw=2):
    figure, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_ylabel('price [USD]', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=16);
    output: BytesIO = io.BytesIO()
    FigureCanvas(figure).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


def train_versus_val_loss(history):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    plt.plot(history.history['loss'], 'r', linewidth=2, label='Train loss')
    plt.plot(history.history['val_loss'], 'g', linewidth=2, label='Validation loss')
    plt.title('LSTM', fontsize=18)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('MSE', fontsize=14)
    ax.legend(loc='best', fontsize=16);
    plt.show()


def viz_predictions(test, preds, targets):
    preds = test[target_col].values[:-window_len] * (preds + 1)
    preds = pd.Series(index=targets.index, data=preds)
    line_plot(targets, preds, 'actual', 'prediction', lw=3)
