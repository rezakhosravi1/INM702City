from abc import ABC, abstractmethod
import numpy as np

class layers(ABC):
    
    def __init__(self,):

        super().__init__()
        pass
    
    @abstractmethod
    def forward(self, X):
        pass

class Linear(layers):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=False):
        
        super().__init__()
        assert type(in_features) is int, 'in_features must be integer' 
        assert type(out_features) is int, 'out_features must be integer' 
        assert in_features > 0, 'in_features must be greater than zero'
        assert out_features > 0, 'out_features must be greater than zero'
        assert type(bias) is bool, 'bias must be boolean'
        self.bias = bias

        if self.bias:
            self.weights = np.random.randn(in_features + 1, out_features)

        else:

            self.weights = np.random.randn(in_features, out_features)

    def forward(self, input):
        if self.bias:
            X_0 = np.ones(input.shape[0])
            X_0 = X_0[:, np.newaxis]
            X = np.concatenate([X_0, input], axis=1)
            return np.matmul(X, self.weights)
        else:
            return np.matmul(input, self.weights)

    
    def backward(self, grad_output):
        grad = grad_output.copy()
        grad_new = np.matmul(grad, self.weights.T)
        return np.clip(grad_new, -1., 1.)