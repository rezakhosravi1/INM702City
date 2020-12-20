# IMPORT LIBRARIES
import numpy as np

# DEFINE BASELINE ALGORITHM FOR THE SHORTEST PATH
class BASP:
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
            returns a suboptimal solution for path and 
            distance (time) that takes for an agent to move 
            from the source to a target.
            Outputs:
            -------
            The first output is the path.
            The second output is the time from the source
            to the target. It is the calculated
            distance minus the initial target value.
        '''
        # PATH FROM THE SOURCE TO THE TARGET
        self.path = {'path':[(0,0)], 'dist':[0.0]}
        current_cell = (0,0)
        termination_condition = True

        # FIND THE NEIGHBOURS (RIGHT OR BOTTOM) OF THE CURRENT CELL 
        while termination_condition:

            neighbors = []
            # THERE IS NO CELL WITH INDEX GREATER THAN THE
            # HEIGHT OR THE WIDTH OF THE GRID
            neighbors.append((current_cell[0],
                                np.min([current_cell[1] + 1, self.width - 1])))
            neighbors.append((np.min([current_cell[0] + 1, self.height - 1]),
                                current_cell[1]))
            # REMOVE THE CURRENT CELL FROM THE NEIGHBOURS SET
            # IF WE ARE ON THE SIDE OR CORNER CELLS OF THE GRID
            # REMOVE THE CELLS WHICH EXCEED THE TARGET LOCATION
            neighbors = list(filter(lambda x: (x!=current_cell and\
                                               x[0] <= self.target[0] and\
                                               x[1] <= self.target[1]), neighbors))

            # FIND THE NEIGHBOUR DISTANCES FROM THE CURRENT CELL
            neighbors_distances = np.array([self.grid[(i,j)] for (i,j) in neighbors])
            # FIND THE BEST NEXT MOVE
            min_distance = np.min(neighbors_distances)
            self.path['dist'] += min_distance
            mask = neighbors_distances == min_distance
            # IF THE NEIGHBOURS HAVE THE SAME DISTACE, CHOOSE ONE OF THEM RANDOMLY
            if len(neighbors_distances[mask]) > 1:
                min_index = np.random.choice(range(2))
                next_cell = neighbors[min_index]
                self.path['path'].append(next_cell)
                current_cell = next_cell
            else:
                next_cell = neighbors[np.argmin(neighbors_distances)]
                self.path['path'].append(next_cell)
                current_cell = next_cell

            # TERMINATE THE LOOP IF THE AGENT REACHES THE TARGET
            if current_cell == self.target:
                termination_condition = False
        
        return self.path['path'], self.path['dist'] - self.grid[-1,-1]