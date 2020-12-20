import numpy as np

class MeanVarNormalize:
    '''
        This class has three methods.
        fit: Takes an array as its argument
        and calculate mean and std for the
        input data.
        transform: Takes an array as its argument
        and subtract the calculated mean from it and
        divide the result by calculated std
        to normalize the input.
        fit_transform:  Joins fit and transform
        methods together.  
    '''

    def __init__(self,):
        pass

    def fit(self, X):
        '''
            Calculates mean and std
            along axis 0.
            Parameters
            ----------
            X: Input array
        '''
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
    def transform(self,X):
        '''
            Normalizes the input array
            and transfer the input data 
            to the new location.
            Parameters
            ----------
            X: Input array
        '''
        X_normalized = (X - self.mean)/self.std
        return X_normalized
    def fit_transform(self, X):
        '''
            Computes mean and std for the input
            array, then, transfer the data to the 
            new location.
            Parameters
            ----------
            X: Input array
            
        '''
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        X_normalized = (X - self.mean)/self.std
        return X_normalized