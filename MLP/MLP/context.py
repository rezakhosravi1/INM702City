from dataclasses import dataclass, field
from typing import Any, Callable

@dataclass
class CTX:
    __slots__ = ['name',
                 'parameters',
                 'activation_inputs',
                 'inputs',
                 'grad_params',
                 'delta']
    name: Any
    parameters: Any
    activation_inputs: Any
    inputs: Any
    grad_params: Any
    delta: Any

    def __repr__(self,):
        return (f'Model: {self.name}\n'
               f'parameters: {self.parameters}')