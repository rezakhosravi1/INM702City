from dataclasses import dataclass, field
from typing import Any
import numpy as np 

@dataclass
class BatchMaker:

    X: np.ndarray
    y: np.ndarray
    n_batches: int
    # shuffle: bool
    cnt: int = 0
    batch_size: int = field(init=False, repr=False)
    X_mini_batches: Any = field(init=False, repr=False)
    y_mini_batches: Any = field(init=False, repr=False)
    
    def __post_init__(self,):
        assert self.n_batches <= self.X.shape[0], 'n_batches overlaps the size' 
        self.batch_size = self.X.shape[0]// self.n_batches
        self.X_mini_batches = []
        self.y_mini_batches = []
        for i in range(self.n_batches):
            if i != self.n_batches-1:
                self.X_mini_batches.append(self.X[i*self.batch_size:(i+1)*\
                                                    self.batch_size, :])
                self.y_mini_batches.append(self.y[i*self.batch_size:(i+1)*\
                                                    self.batch_size, :])
            else:
                self.X_mini_batches.append(self.X[i*self.batch_size:, :])
                self.y_mini_batches.append(self.y[i*self.batch_size:, :])

    def __len__(self,):
        return len(self.X_mini_batches[0])
    
    def __iter__(self,):
        self.cnt = 0
        return self

    def __next__(self,):
        if self.cnt < len(self.X_mini_batches):
            cnt_ = self.cnt
            self.cnt += 1
            return zip([self.X_mini_batches[cnt_], self.y_mini_batches[cnt_]])
        else:
            raise StopIteration