# -*- coding: utf-8 -*-

def prescale_whole_matrix(data_mtx,prescaler):
    """
    Function to apply sklearn preprocessing to a matrix as a whole, rather than each feature independently
    data_mtx would be a numpy matrix that you would input into the prescaler
    prescaler would be an sklearn prescaler, like StandardScaler() for example
    """
    return prescaler.fit_transform(data_mtx.ravel().reshape(-1, 1)).reshape(data_mtx.shape), prescaler