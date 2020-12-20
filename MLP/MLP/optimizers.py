import numpy as np 

class Optimizer(object):

    def __new__(cls,
                name,
                net_parameters,
                lr):
        '''
            net_parameters is ctx
        '''
        assert name in ['sgd'], 'Invalid optimizer'
        if name == 'sgd':
            return SGD(net_parameters,lr)        

class SGD:

    def __init__(self,
                 ctx,
                 lr):
        self.lr = lr
        self.ctx = ctx

    def step(self,):
        for i in range(len(self.ctx.grad_params)):
            self.ctx.parameters[i] -= np.full_like(self.ctx.grad_params[i], self.lr) * self.ctx.grad_params[i]
    
    def zero_grad(self,):
        self.ctx.grad_params = []