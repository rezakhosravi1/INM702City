import numpy as np 

class Loss(object):

    def __new__(cls, name):
        assert name in ['mse',
                        'bce',
                        'cce',
                        'softmax_cce'], 'Invalid loss'
        if name == 'mse':
            return MeanSquareErrorLoss()
        elif name == 'bce':
            return BinaryCrossEntropyLoss()
        elif name == 'cce':
            return CategoricalCrossEntropyLoss()
        elif name == 'softmax_cce':
            return CategoricalCrossEntropyWithSoftmaxTogether()


class MeanSquareErrorLoss:
    
    def calc(self, y_eval, y_true):
        return np.sum(0.5*(y_eval - y_true)**2)/len(y_true)
    
    def gradient(self, y_eval, y_true):
        grad = y_eval - y_true 
        return np.clip(grad, -1., 1.)

class BinaryCrossEntropyLoss:
    
    def calc(self, y_eval, y_true):
        return -1./len(y_true) * np.sum((y_true*np.log(y_eval) +\
                                        (1-y_true)*np.log(1-y_eval + 1e-5)))
    
    def gradient(self, y_eval, y_true):
        grad = (y_eval - y_true)/(y_eval * (1 - y_eval))
        return np.clip(grad, -1., 1.)

class CategoricalCrossEntropyLoss:
    
    def calc(self, y_eval, y_true):
        return -1./len(y_true) * \
            np.einsum('ii', np.einsum('ij,kj', y_true, np.log(y_eval + 1e-5)))
    
    def gradient(self, y_eval, y_true):
        grad = - y_true / (y_eval + 1e-5)
        return np.clip(grad, -1., 1.)

class CategoricalCrossEntropyWithSoftmaxTogether:
    
    def calc(self, y_eval, y_true):
        return -1./len(y_true) * \
            np.einsum('ii', np.einsum('ij,kj', y_true, np.log(y_eval + 1e-5)))
    
    def gradient(self, y_eval, y_true):
        grad = y_eval - y_true

        return np.clip(grad, -1., 1.)