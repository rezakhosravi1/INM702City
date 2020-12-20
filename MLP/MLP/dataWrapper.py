from dataclasses import dataclass, field
from typing import Any, Callable

@dataclass
class DataWrapper:

    __slots__ = ['data',
                 'function']

    data: Any
    function: Callable

    def __iter__(self,):
        batches = iter(self.data)
        for batch in batches:
            if self.function == None:
                yield batch
            else:
                yield self.function(*batch)