import numpy as np 

def train_test_split(X, y, test_ratio=0.2):
    '''
        test_ratio: The ratio of sample 
        test data size. 
        valid_ratio: The ratio of sample
        validation data size.
        Parameters
        ----------
        X: Data points , y: labels
        test_ratio: ratio of test data to
        the whole dataset.
        valid_ratio: ratio of valid data to
        the wholde dataset.
    '''
    
    assert len(X) == len(y)
    p = np.random.permutation(len(X))  # SHUFFLE DATA
    X = X[p]
    y = y[p]
    y = y[:, np.newaxis]  # MAKE 1D ARRAY TO 2D FOR CALCULATION PURPOSES

    X_test, X_train = np.split(X,
                               [int(len(X)*test_ratio)])
    y_test, y_train = np.split(y,
                               [int(len(y)*test_ratio)])

    return X_train, X_test, y_train, y_test