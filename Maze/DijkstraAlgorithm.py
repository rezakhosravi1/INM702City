# IMPORT LIBRARIES
import numpy as np
from copy import deepcopy

# DEFINE DIJKSTRA'S ALGORITHM
class Dijkstra:
    '''
        This class gets a grid and a target location
        as its arguments and returns the path and time 
        of the shortest path from location(0,0) to the
        target as its outputs.
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
        
        # UNVISITED CELLS WHICH THE AGENT HAS NOT FOUND THE MIN DIST TO THEM 
        self.unvisited_cells = [(i,j) for i in range(self.height) \
                                      for j in range(self.width)]
        # MIN PATH SEQUENCE FROM SOURCE TO EACH CELL
        self.shortest_paths ={str(unvisited_cell):[] \
                              for unvisited_cell in self.unvisited_cells}
        # THE MIN DISTANCE FROM SOURCE TO EACH CELL
        # FIRST, WE ASSIGN INFINIT DISTANCE TO ALL CELLS EXCEPT SOURCE
        # WE ASSING ZERO TO SOURCE CELL  
        self.distance = np.full((self.height, self.width), np.inf)
        # ASSIGN ZERO TO THE SOURCE DISTANCE
        # BECAUSE IT HAS NO EFFECT ON THE FINAL RESULT 
        self.distance[0,0] = 0

    def find_the_shortest_path(self,):
        '''
            This function returns the shortest path and 
            distance between the source and the target.
            Outputs:
            -------
            The first output is all the shortest distances 
            between source and all visited cells and is 
            the tentative distance for the unvisited cells.
            The second output is the shortest time from
            the source to the target. It is the calculated
            distance minus the initial target value .
            The third output is the path from the source to 
            the target. 
        '''
        # THE CURRENT CELL TO CALCULATE AND FIND THE MIN DISTANCE
        current_cell = (0,0)
        # DEFINE A COPY OF DISTANCE FOR NEXT STEP
        # TO FIND THE NEXT MINIMUM DISTANCE CELL
        min_dist = self.distance.copy()

        # IF THE TARGET IS A VISITED NODE THEN WE HAVE FOUND
        # THE SHORTEST PATH TO IT, REAGRDING THIS WE TERMINATE
        # THE COMPUTATION
        while (self.target in self.unvisited_cells):

            # FINDING THE NEIGHBORS TO THE CURRENT CELL
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
            neighbors = list(filter(lambda x: (x!=current_cell and\
                                               x in self.unvisited_cells), neighbors))
            # THE DISTANCE FROM CURRENT NODE TO ALL ITS NEIGHBORS
            # WHICH ARE NOT VISITED NODES
            for (i,j) in neighbors:

                # CHOOSING BETWEEN THE TENTATIVE DISTANCE THAT THE NEIGHBORS
                # CURRENTLY HAVE OR THE SUM OF THE CURRENT CELL TO THE ORIGIN
                # AND THE NUMBER ON THE CELL
                # ADD THE CELLS TO THE PATH OF THE NEIGHBOR CELLS IF
                # THIS PATH IS SHORTER THE PREVIOUS PATH
                if (self.distance[current_cell] + self.grid[(i,j)] < \
                self.distance[(i,j)]):

                    self.distance[(i,j)] = self.distance[current_cell] +\
                                            self.grid[(i,j)]
                    # 
                    min_dist[(i,j)] = self.distance[current_cell] +\
                                            self.grid[(i,j)]
                    # USE DEEPCOPY TO AVOID THE CHANGING THE PATH FOR
                    # PARENT CELLS
                    self.shortest_paths[str((i,j))] = \
                            deepcopy(self.shortest_paths[str(current_cell)])
                    self.shortest_paths[str((i,j))].append(current_cell)

            # REMOVE THE CURRENT CELL FORM THE UNVISITED CELLS AND MIN_DIST
            self.unvisited_cells.remove(current_cell)
            # TO REMOVE THE CURRENT CELL FROM MIN_DIST
            # WE ASSIGN INFINITY TO THIS CELL
            min_dist[current_cell] = np.inf
            # MAKE THE MINIMUM DISTANCE CELL AS THE NEW CURRENT CELL
            # THERE MIGHT BE SEVERAL MINIMUM DISTANCES SO PEAK ONE
            # HERE WE CHOOSE THE FIRST INDEX
            current_cell = np.argwhere(min_dist == np.min(min_dist))[0]
            # CHANGING FROM ARRAY TYPE TO TUPLE TYPE
            current_cell = (current_cell[0], current_cell[1])
        # ADD EACH CELL INDEX TO ITS SHORTEST PATH FROM THE SOURCE
        for i in range(self.height):
            for j in range(self.width):
                self.shortest_paths[str((i,j))].append((i,j))
        
        return self.distance,\
               self.distance[-1,-1] - self.grid[-1,-1],\
               self.shortest_paths[str(self.target)]