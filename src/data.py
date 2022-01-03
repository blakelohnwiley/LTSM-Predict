# import packages
from json import loads

import numpy as np
import pandas as pd
import requests

from static import successful_response_codes, window_len, zero_base, test_size, target_col

"""
Author: Blake Lohn-Wiley (maths.lohnwiley@gmail.com)
Date: 03.01.2022
"""

class Data:
    """Class for obtaining the data used for training and testing the model"""

    def __init__(self, method, endpoint, params):
        """
        Get the data from the provided api url and path parameters
        :param method: HTTP Verb (GET,POST,POST,PATCH,DELETE)
        :param endpoint: Full endpoint url.
        :param params: Path parameters to pass
        """
        self.method = method
        self.url = endpoint
        self.path_params = params

    def get_data(self):
        """Make the http request to get the data, return as a
        raw json response."""
        # make the request
        http_request = requests.request(method=self.method,
                                        url=self.url,
                                        params=self.path_params)
        try:
            # provide response logic in case the request fails.
            if http_request.status_code in successful_response_codes:
                # return the response if successful
                res_content = http_request.content.decode('utf-8')
                # convert into json dictionary -- easier to manipulate this way
                return loads(res_content)

        except Exception as e:
            """inform end user of problem by printing out to console -- 
            dumping the response content. An api error is not always contented within
            the json body. Need to use content and decode."""
            error_message = http_request.content.decode('utf-8')
            raise Exception({"error": error_message})


class FormatData:
    """
    Class with methods to format data into training and testing
    datasets.
    """

    def __init__(self, data):
        """
        Data to be formatted and prepared for training and testing.
        """
        self.data = data
        self.target_col = "close"
        self.zero_base = zero_base
        self.test_size = test_size
        self.window_len = window_len
        self.test_size = test_size

    def normalise_zero_base(self, df):
        """Normalize zero base"""
        return df / df.iloc[0] - 1

    def format_data(self):
        """Format the data to ensure that it's formatted to be
        used in training and testing."""
        # variable that will be predicting
        formatted_data = pd.DataFrame(self.data['Data'])
        formatted_data = formatted_data.set_index('time')
        formatted_data.index = pd.to_datetime(formatted_data.index, unit='s')
        # drop columns, set axes and hold emplace true.
        formatted_data.drop(["conversionType", "conversionSymbol"], axis='columns', inplace=True)
        # return data frame
        return formatted_data

    def train_test_split(self, df, test_size):
        """Split the data into training and test data."""
        split_row = len(df) - int(test_size * len(df))
        train_data = df.iloc[:split_row]
        test_data = df.iloc[split_row:]
        return train_data, test_data

    def extract_window_data(self, df, window_len, zero_base):
        window_data = []
        for idx in range(len(df) - window_len):
            tmp = df[idx: (idx + window_len)].copy()
            if zero_base:
                tmp = self.normalise_zero_base(tmp)
            window_data.append(tmp.values)
        return np.array(window_data)

    def prepare_data(self, df, target_col, window_len, zero_base, test_size):
        """Split the data into training and test data."""
        train_data, test_data = self.train_test_split(df, test_size)
        X_train = self.extract_window_data(train_data, window_len, zero_base)
        X_test = self.extract_window_data(test_data, window_len, zero_base)
        y_train = train_data[target_col][window_len:].values
        y_test = test_data[target_col][window_len:].values
        if zero_base:
            y_train = y_train / train_data[target_col][:-window_len].values - 1
            y_test = y_test / test_data[target_col][:-window_len].values - 1

        return train_data, test_data, X_train, X_test, y_train, y_test


# params = {
#     "fsym":"BTC",
#     "tsym":"USD",
#     "limit":500
# }
# method = "GET"
# full_url = "https://min-api.cryptocompare.com/data/histominute"
# unformatted_data = Data(method,full_url,params).get_data()
# formatted_df = FormatData(unformatted_data).format_data()
#
# train, test, X_train, X_test, y_train, y_test = FormatData(unformatted_data).prepare_data(
#     formatted_df,target_col,window_len,zero_base,test_size)
