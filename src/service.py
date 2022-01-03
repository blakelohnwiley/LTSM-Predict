import datetime
import os
import sys

sys.path.insert(1, os.path.join(os.getcwd(), "../src/models"))

from dynamic_model import DynamicModel
from tensorflow.keras.models import save_model
from static import (full_url, params, method, target_col,
                    window_len, zero_base,
                    test_size, output_size, neurons, dropout,
                    loss, optimizer, epochs, batch_size,verbose, shuffle,
                    activ_func)
from data import Data, FormatData


"""
Author: Blake Lohn-Wiley (maths.lohnwiley@gmail.com)
Date: 03.01.2022
"""


class Service:
    # model_name must be supplied.
    # otherwise no configuration cad be loaded.
    def __init__(self, model_name=None):
        self.model_name = model_name
        self.dynamic_model = DynamicModel(self.model_name)
        self.target_col = target_col

    def _train(self):
        """Train the model with training data"""
        self.training_data = Data(method, full_url, params).get_data()
        self.formatted_df = FormatData(self.training_data).format_data()
        # split up into x-test,train,y-test,train
        self.train, self.test, self.X_train, self.X_test, self.y_train, self.y_test = FormatData(
            self.training_data).prepare_data(self.formatted_df,
                                             target_col,
                                             window_len,
                                             zero_base,
                                             test_size)

        # This return the compiled Keras Model from dynamic_model->model()
        model = self.dynamic_model.build_lstm_model(self.X_train,
                                                    output_size,
                                                    neurons,
                                                    activ_func,
                                                    dropout,
                                                    loss,
                                                    optimizer)

        # fit the model based off of training and testing data
        model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=epochs, batch_size=batch_size, verbose=verbose,
            shuffle=shuffle)

        # Save trained model
        MODEL_DIR = "../src/models/"
        now = datetime.datetime.now()
        version = 1
        export_path = os.path.join(MODEL_DIR, str(version))
        print('export_path = {}\n'.format(export_path))
        model.save(
            filepath=f"{export_path}/{self.model_name}_{now.year}{now.month}{now.day}_{now.hour}{now.minute}",
            overwrite=True,
            include_optimizer=True,
            save_format=None,
            signatures=None,
            options=None
        )
        return True

    #TODO:
    # Implement predict via restful API

    # def predict(self, X):
    #     # Load model
    #     model = load_model("models/ModelA")
    #
    #     # Execute
    #     results = model.predict(X)
    #     if results is not None and results != False:
    #         return results
    #     return False
