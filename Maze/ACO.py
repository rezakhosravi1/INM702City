# IMPORT LIBRARIES
import numpy as np
from copy import deepcopy

# DEFINE ACS
class ACS:
    '''
        This class gets a grid and a target location
        as its arguments and returns the path and time 
        of the shortest path from location(0,0) to the
        target as its outputs. Note: The convergence depends
        on the number of runs and parameters.
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

        self.eta = 1./(self.grid + 1. )  # TO AVOID INFINITY  
        self.eta[0,0] = 0.
        
        assert type(target) is tuple, 'target must be a tuple'
        assert len(target) == 2, 'target length must be of 2'
        assert target[0] <= self.height, 'target first element must be <= grid height'
        assert target[1] <= self.width, 'target second element must be <= grid width'
        assert target[0] >= 1, 'target elements must be positive'
        assert target[1] >= 1, 'target elements must be positive'
        self.target = (target[0]-1,target[1]-1)

    def find_the_shortest_path(
        self,
        n_ants=10, n_iterations=100,
        alpha=1, beta=1.3,
        tau_0=1, phi=0.1, rho=0.1, initial_prob=0.9,
        callbacks=None
        ):
        '''
            This function gets number of ants and
            number of iterations as its arguments and
            returns the shortest path and the time that
            takes to reach the target.
            Parameters
            ----------
            n_ants: Integer input that defines the number of 
            artificial ants
            n_iterations: Integer input that defines the number
            of iterations.
            alpha: Parameter to control the impact of pheromone.
            beta:  Parameter to control the impact of heuristic
            imformation.
            tau_0: Initial pheromone level
            phi: Decay parameter for local update
            rho: Decay parameter for global update
            initial_prob: Initial probability to choose the 
            next vertex based ACS paper.
            callbacks: A list consist which moniter the 
            loss and if there is no improvement after a certain
            number of iterations then terminates the process.
            The first element is the type of loss that we want to 
            monitor. The second element is a threshold. The third 
            element is the number of patience.
        '''
        assert type(n_ants) is int, 'n_ants must be an integer'
        assert type(n_iterations) is int, 'n_iterations must be an integer'  
        assert n_ants >= 1, 'n_ants must be greater than 1'
        assert n_iterations >= 1, 'n_iterations must be greater than 1'

        assert alpha >= 0, 'alpha must be positive'
        self.alpha = alpha

        assert beta >= 1, 'beta must be greater than 1'
        self.beta = beta

        assert tau_0 >= 0, 'tau_0 must be positive'
        self.tau_0 = tau_0

        assert (phi >= 0 and phi <= 1), 'phi must be between 0 and 1'
        self.phi = phi

        assert (rho >= 0 and rho <= 1), 'rho must be between 0 and 1'
        self.rho = rho

        assert (initial_prob >= 0 and \
            initial_prob <= 1), 'initial_prob must be between 0 and 1'
        self.initial_prob = initial_prob

        self.tau = np.full((self.height, self.width), 1e-5)
        # TO GIVE THE HIGHER CHANCE IF AN ANT IS ON A TARGET NEIGHBOR CELL
        self.tau[self.target] = 1e5  
        self.tau[(0,0)] = 0.  # AVOID TO MOVE TO THE SOURCE

        # A COUNTER TO TERMINATE THE PROCESS IF THERE IS NO IMPROVEMENT 
        self.patience_counter = 0

        # TO FIND THE BEST ANT UP TO THE CURRENT ITERATION
        self.best_path = {'path':[], 'dist':[]}
        
        # A TERMINAL CONDITION FO
        self.termination_condition = True

        while self.termination_condition:

            for iteration in range(n_iterations):
                
                # TO FIND THE BEST ANT ON EACH ITERATION
                distances = []
                # PATH OF EACH ANT FROM SOURCE TO TARGET IN THE CURRENT ITERATION
                self.paths = { str(i):{'path':[],'dist':[0.]} for i in range(n_ants) }
                
                # TO UPDATE PHROMONE BASED ON THE PATH OF THE BEST ANT
                self.delta_tau_best = np.zeros_like(self.tau)


                for n_ant in range(n_ants):
                    # DEFINE THE SOURCE
                    current_cell = (0,0)
                    # ADD THE SOURCE TO THE CURRENT ANT OF THE CURRENT ITERATION 
                    self.paths[str(n_ant)]['path'].append(current_cell)
                    # PUT ALL THE CELL IN A SET OF UNVISITED CELLS
                    self.unvisited_cells = [(i,j) for i in range(self.height) \
                                                for j in range(self.width)]
                    next_cell = (0,0)  # LOCAL VARIABLE FOR THE NEXT MOVE

                    while self.target in self.unvisited_cells:

                        try:
                            # FIND THE CURRENT CELL FEASIBLE NEIGHBORS
                            neighbors, neighbors_prob, num_max_probs = self.neighbors(current_cell)
                            
                            # GENERATE A RANDOM NUMBER FROM UNIFORM DISTRIBUTION
                            # IS IT IS LESS THAN THE INITIAL PROBABILITY THEN CHOOSE
                            # THE CELL WHICH MAXIMIZES TAU(I,J) * ETA(I,J)
                            if (np.random.rand(1) < self.initial_prob):
                                
                                # IF WE HAVE MORE THAN ONE NEIGHBOR WITH
                                # THE HIGHEST PROBABILITY 
                                if num_max_probs > 1:
                                    
                                    # ASSIGN ZERO PROBABILITY TO NON-MAXIMUM ELEMENTS
                                    neighbors_prob[neighbors_prob!=np.max(neighbors_prob)] = 0.
                                    # TO MAKE THE NEIGHBORS_PROB A PROBABILITY DISTRIBUTION 
                                    neighbors_prob = neighbors_prob / np.sum(neighbors_prob)
                                    next_cell_index = int(np.random.choice(range(len(neighbors_prob)),
                                                                            size=(1,),
                                                                            p=neighbors_prob))
                                    next_cell = neighbors[next_cell_index]

                                else:
                                    
                                    next_cell = neighbors[np.argmax(neighbors_prob)]

                            else:

                                next_cell_index = int(np.random.choice(range(len(neighbors_prob)),
                                                                        size=(1,),
                                                                        p=neighbors_prob))
                                next_cell = neighbors[next_cell_index]
                        
                        # PRINT ERROR IF THERE IS ANY NAN PROBABILITY 
                        except ValueError:

                            print(current_cell,'\n')
                            print('There is a nan probability element in neighbors_prob ')
        
                        # LOCAL UPDATE
                        current_cell = self.local_update(n_ant, current_cell,next_cell)
                        # TERMINATE THE CURRENT ANT PROCESS IF IT REACHES THE TARGET
                        if current_cell == self.target:
                            break

                    # ADD THE CURRENT ANT DISTANCE TO THE LIST
                    distances.append(self.paths[str(n_ant)]['dist'])

                # GLOBAL UPDATE
                self.global_update(distances)

                # CHECK IF THERE IS NO IMPROVEMENT 
                if callbacks != None:
                    
                    self.callback(callbacks, distances)

                    if (self.patience_counter == callbacks[2] or \
                    iteration == (n_iterations - 1)):
                        
                        self.termination_condition = False
                        break
                
                if iteration == n_iterations - 1:
                    
                    self.termination_condition = False
                    break

        return self.best_path['path'], self.best_path['dist']


    def neighbors(self, current_cell):
        '''
            Returns the current cell feasible neighbors,
            their probablities and the number of neighbors
            with the maximum probability.
        '''
        # FINDING THE NEIGHBORS TO THE CURRENT CELL
        neighbors = []
        neighbors_prob = []
        num_max_probs = 1
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
        # REMOVE THE VISITED CELLS
        neighbors = list(filter(lambda x: (x!=current_cell and \
                                        x in self.unvisited_cells),
                                neighbors))

        if len(neighbors) > 0:

            for (i,j) in neighbors:
                
                neighbors_prob.append((self.tau[(i,j)] ** self.alpha) * \
                                        (self.eta[(i,j)] ** self.beta))
            
            neighbors_prob = np.array(neighbors_prob)
            neighbors_prob = neighbors_prob / np.sum(neighbors_prob)
            
            # # TO SOLVE THE INFINITY DIVISION BY INFINITY
            # if sum(np.isnan(neighbors_prob)):
            #     neighbors_prob[np.isnan(neighbors_prob)] = 1.
            #     neighbors_prob = neighbors_prob / np.sum(neighbors_prob)
            
            num_max_probs = sum(neighbors_prob==np.max(neighbors_prob))
            
        # IF ALL NEIGHBORS ARE VISITED CHOOSE RANDOM ONE
        else:

            neighbors = []
            neighbors.append((current_cell[0],
                            np.max([current_cell[1] - 1, 0])))
            neighbors.append((np.max([current_cell[0] - 1, 0]),
                            current_cell[1]))
            neighbors.append((current_cell[0],
                            np.min([current_cell[1] + 1, self.width - 1])))
            neighbors.append((np.min([current_cell[0] + 1, self.height - 1]),
                            current_cell[1]))
            neighbors = list(filter(lambda x: x!=current_cell, neighbors))

            # UNIFORM PROBABILITIES
            neighbors_prob = np.full((len(neighbors),), 1./len(neighbors))
            
            num_max_probs = sum(neighbors_prob==np.max(neighbors_prob))
        
        return neighbors, neighbors_prob, num_max_probs

    
    def local_update(self, n_ant, current_cell,next_cell):
        '''
            Updates the phromone level and returns the 
            next cell as the new current cell.
        '''
        
        self.paths[str(n_ant)]['path'].append(next_cell)
        self.paths[str(n_ant)]['dist'] += self.grid[next_cell]
        # LOCAL UPDATE 
        self.tau[next_cell] = \
                              (1 - self.phi) * self.tau[next_cell] + \
                              self.phi * self.tau_0
        
        # MARK THE CURRENT CELL AS VISITED AND UPDATE THE NEW ONE
        if current_cell in self.unvisited_cells:
            self.unvisited_cells.remove(current_cell)
        
        return next_cell


    def global_update(self, distances):
        '''
            Gets the distance of all ants for the current 
            iteration and updates the best ant so far.
        '''

        best_ant = np.argmin(distances)
        if self.best_path['path']:
            if distances[best_ant] < self.best_path['dist']:
                self.best_path['path'] = deepcopy(
                                        self.paths[str(best_ant)]['path'])
                self.best_path['dist'] = deepcopy(
                                        self.paths[str(best_ant)]['dist'])
        else:
            self.best_path['path'] = deepcopy(
                                        self.paths[str(best_ant)]['path'])
            self.best_path['dist'] = deepcopy(
                                        self.paths[str(best_ant)]['dist'])

        for (i,j) in self.best_path['path']:
            self.delta_tau_best[(i,j)] = 1./deepcopy(self.best_path['dist'])
            self.delta_tau_best[(0,0)] = 0.

        self.tau = (1.-self.rho) * self.tau + self.delta_tau_best

    
    def loss(self, distances, loss):
        '''
            The criterion rule for checking the improvement.
        '''
        assert loss in ['absolute_difference'], 'Undefined loss type'
        if loss == 'absolute_difference':
            return abs(deepcopy(self.best_path['dist']) - np.min(distances))

    def callback(self, callbacks, distances):
        '''
            Checks whether there is improvement or not.
        '''

        if self.loss(distances, callbacks[0]) < callbacks[1]:
            self.patience_counter += 1
        else:
            self.patience_counter = 0