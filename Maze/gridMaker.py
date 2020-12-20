# IMPORT LIBRARIES
import numpy as np

# DEFINE THE GRID 
class Grid:
    '''
        This class gets the height and width as 
        its arguments and returns a grid of this size 
        with random digits(0-9).
        Parameters
        ----------
        height: Input positive integer number 
        width: Input positive integer number

    '''

    def __init__(self, height, width):
        
        assert type(height) is int, 'height must be an integer'
        assert height > 0, 'height must be positive'

        assert type(width) is int, 'width must be an integer'
        assert width > 0, 'width must be positive'

        self.height = height
        self.width = width

    def make_grid(self,):
        '''
            returns a grid of size (height,width) with
            random digits(0-9)
        '''
        return np.random.randint(0, 10, size=(self.height,self.width))