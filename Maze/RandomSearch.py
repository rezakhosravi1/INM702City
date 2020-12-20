# IMPORT LIBRARIES
import numpy as np

# DEFINE THE RANDOM SEARCH CLASS
class RandomSearch:
    '''
        This class gets a grid and a target location
        as its arguments and returns the path and time 
        as its outputs.
        Prameters
        ---------
        grid: Input array, sample grid from the ouput of make_grid
        method of a Grid class object.
        target: Input tuple, the location of the target if we assume
        that the source location is cell (1,1). 
    '''

    def __init__(self, grid, target):

        assert type(grid) is np.ndarray, 'grid must be a numpy array'
        assert grid.shape[0] > 1, 'grid first dim must be greater than zero'
        assert grid.shape[1] > 1, 'grid second dim must be greater than zero'
        assert all(isinstance(x, np.int32) \
               for x in grid.ravel()), 'grid elements must be integers'
        assert grid.all() < 10, 'grid elements must be digits(0-9)'
        
        self.grid = grid
        self.height = grid.shape[0]
        self.width = grid.shape[1]


        assert type(target) is tuple, 'target must be a tuple'
        assert len(target) == 2, 'target length must be of 2'
        assert target[0] <= self.height, 'target first element must be <= grid height'
        assert target[1] <= self.width, 'target second element must be <= grid width'
        assert target[0] >= 1, 'target elements must be positive'
        assert target[1] >= 1, 'target elements must be positive'

        self.target = (target[0]-1,target[1]-1)
        
    def find_the_path(self, ):
        '''
            returns path and distance (time) that takes
            for an agent to walk with random decision
            from the source to a target.
        '''

        # PATH FROM THE SOURCE TO THE TARGET
        self.path = {'path':[(0,0)], 'dist':[0.0]}
        current_cell = (0,0)
        termination_condition = True

        # FIND THE FEASIBLE NEIGHBOURS AND CHOOSE THE LOWEST DISTANCE 
        while termination_condition:
            
            neighbors = []
            # THERE IS NO CELL WITH NEGATIVE INDEX
            neighbors.append((current_cell[0],
                              np.max([current_cell[1] - 1, 0])))
            neighbors.append((np.max([current_cell[0] - 1, 0]),
                              current_cell[1]))
            # THERE IS NO CELL WITH INDEX GREATER THAN THE
            # HEIGHT OR THE WIDTH OF THE GRID
            neighbors.append((current_cell[0],
                              np.min([current_cell[1] + 1, self.width - 1])))
            neighbors.append((np.min([current_cell[0] + 1, self.height - 1]),
                              current_cell[1]))
            # REMOVE THE CURRENT CELL FROM THE NEIGHBOURS SET
            # IF WE ARE ON THE SIDE OR CORNER CELLS OF THE GRID
            neighbors = list(filter(lambda x: x!=current_cell, neighbors))
            
            # CHOOSE RANDOMLY BETWEEN THE NEIGHBOUR CELLS 
            next_cell = neighbors[np.random.choice(range(len(neighbors)))]
            # ADD THE CELL LOCATION TO PATH
            self.path['path'].append(next_cell)
            # ADD ITS EDGE TO DIST
            self.path['dist'] += self.grid[next_cell]
            # UPDATE THE CURRENT CELL
            current_cell = next_cell

            # TERMINATE THE LOOP IF THE AGENT REACHES THE TARGET
            if current_cell == self.target:
                termination_condition = False
            
        return self.path['dist'] - self.grid[-1,-1], self.path['path']