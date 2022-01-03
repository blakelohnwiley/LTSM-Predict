# import packages
import numpy as np

"""
Author: Blake Lohn-Wiley (maths.lohnwiley@gmail.com)
Date: 03.01.2022
"""

seed = np.random.seed(42)
output_size=1
target_col = 'close'
window_len = 5
test_size = 0.2
zero_base = True
neurons = 100
epochs = 20
batch_size = 32
loss = 'mse'
dropout = 0.2
optimizer = 'adam'
activ_func = 'linear'
verbose = 1
shuffle = True
successful_response_codes = list(range(200,299))
method = "GET"
full_url = "https://min-api.cryptocompare.com/data/histominute"
params = {
    "fsym":"BTC",
    "tsym":"USD",
    "limit":500
}
