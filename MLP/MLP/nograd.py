import numpy as np
from copy import deepcopy
import functools
import inspect


class NoGrad:

    def __new__(cls, obj):
        if inspect.isfunction(obj):
            return ReturnNoGrad(obj)
        else:
            instance = ReturnNoGrad(obj)
            return instance.get_object()

class ReturnNoGrad:

    def __init__(self, obj):
        if inspect.isfunction(obj):
            functools.update_wrapper(self, obj)
        self.object = deepcopy(obj)

        if hasattr(self.object, 'requires_grad'):
            self.object.requires_grad = False

    def get_object(self,):
        return self.object

    def __call__(self, *args, **kwargs):
        # GET ALL ARGS AND KWARGS WHICH IS THE INDEX 1 OF
        # THE FUNCTION GETFULLARGSSPEC
        # FROM INSPECT BUILT-IN PYTHON LIBRARY 
        args_kwargs = inspect.getfullargspec(self.object)[0]
        if 'requires_grad' in args_kwargs:
            return self.object(*args, **kwargs, requires_grad=False)
        else:
            return self.object(*args, **kwargs)