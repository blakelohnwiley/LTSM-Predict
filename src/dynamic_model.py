# import packages
from tensorflow.keras.layers import Activation, Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from static import dropout, neurons, activ_func, optimizer, loss

"""
Author: Blake Lohn-Wiley (maths.lohnwiley@gmail.com)
Date: 03.01.2022
"""


class DynamicModel:
    """
    Generate the dynamic LTSM model with attrs
    """

    def __init__(self, model_name=None, input_data=None, output_size=None, neurons=neurons, activ_func=activ_func,
                 dropout=dropout, loss=loss, optimizer=optimizer):
        """
        Initialize the model.
        :param model_name: Name of the model.
        :param input_data: Input dataset set used for training and testing
        :param output_size: Output size
        :param neurons: Number of neurons
        :param activ_func: activ function.
        :param dropout: The dropout rate.
        :param loss:  The loss rate
        :param optimizer: optimizer to use
        """
        self.model_name = model_name
        self.input_data = input_data
        self.output_size = output_size
        self.neurons = neurons
        self.activ_func = activ_func
        self.dropout = dropout
        self.loss = loss
        self.optimizer = optimizer

    def build_lstm_model(self, input_data, output_size, neurons, activ_func, dropout, loss, optimizer):
        model = Sequential()
        model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
        model.add(Dropout(dropout))
        model.add(Dense(units=output_size))
        model.add(Activation(activ_func))
        # Compile the model
        model.compile(loss=loss, optimizer=optimizer)
        return model
