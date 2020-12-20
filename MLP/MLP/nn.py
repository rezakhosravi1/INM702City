import numpy as np 
from sklearn.datasets import make_blobs
import plotly.graph_objects as go
import sys
from time import sleep
from dataclasses import dataclass, field
from typing import Any, Callable
from abc import ABC, abstractmethod
import numpy as np
from copy import deepcopy
import functools
import inspect
from MLP.nograd import NoGrad
from MLP.context import CTX
from MLP import layers
from MLP import losses
from MLP.dataWrapper import DataWrapper
from MLP.batchMaker import BatchMaker
from MLP import activations
from MLP import optimizers


class Model:

    def __init__(self,
                 name):
        
        self.ctx = CTX(name=name,
                       parameters=[],
                       activation_inputs=[],
                       inputs=[],
                       grad_params=[],
                       delta=[])
        

    def parameters(self,):
        return self.ctx
    
    def name_layers(self, ):
        return [attribute for attribute in \
                self.__dict__.keys() \
                if (attribute[:2] !='__' and \
                attribute not in ['ctx'])]
    
    def compile(self, layers):
        assert type(layers) is list, 'layers must be a list'
        assert all(isinstance(x, str)
               for x in layers), 'layers elements must be string'
        assert all(x in self.name_layers() \
               for x in layers), 'Invalid layers'
        
        self.layers = layers
        
        for layer in layers[1:]:
            attribute = getattr(self, layer)
            if hasattr(attribute, 'bias'):
                if attribute.bias == True:
                    raise AttributeError('Hidden layers other than first layer' +
                                         ' cannot have bias')

        for name in self.layers:
            net_attribute = getattr(self, name)
            if hasattr(net_attribute, 'weights'):
                self.ctx.parameters.append(net_attribute.weights)
        
    def forward(self, input):
        attribute1 = getattr(self, self.layers[0])
        if attribute1.bias:
            X_0 = np.ones(input.shape[0])
            X_0 = X_0[:, np.newaxis]
            X = np.concatenate([X_0, input],axis=1)
            self.ctx.inputs.append(X)
        else:
            self.ctx.inputs.append(input)
        
        for layer in self.layers:
            attribute = getattr(self, layer)
            if hasattr(attribute, 'weights'):
                input = attribute.forward(input)
            else:
                input = attribute.forward(self.ctx, input)
        
        # if len(input.shape) == 1:
            # input = input[:, np.newaxis]
        
        return input
    
    def backward(self,
                 loss):
                 
        grad_output = loss.copy()
        
        for layer in self.layers[::-1]:  # REVERSE COPY OF LAYERS NAME
            attribute = getattr(self, layer)
            if hasattr(attribute, 'weights'):
                self.ctx.delta.insert(0, grad_output)
                grad_output = attribute.backward(grad_output)
            else:
                grad_output = attribute.backward(self.ctx, grad_output)
        
        # CALCULATE THE GRADIENT OF WEIGHTS
        for i in range(len(self.ctx.delta)):
            # if len(self.ctx.inputs[i].shape) == 1:
                # self.ctx.inputs[i] = self.ctx[:, np.newaxis]
            self.ctx.grad_params.append(np.matmul(self.ctx.inputs[i].T, self.ctx.delta[i]))
        # RESET THE INPUT OF EACH LAYER FOR NEXT ITERATION
        self.ctx.delta = []
        self.ctx.inputs = []
    
    def fit(self,
            X_train,
            y_train,
            optimizer,
            loss,
            n_epochs,
            n_batches=1,
            shuffle=True,
            valid_ratio=None,
            n_valid_bathces=1,
            preprocessing=None,
            verbose=True
            ):

        self.loss = loss
        self.history = dict(train_loss=[], val_loss=[])
        
        self.X = X_train
        self.y = y_train
        
        if valid_ratio:
            
            self.X_valid, self.X_train = np.split(self.X,
                                                 [int(len(self.X)*valid_ratio)])
            self.y_valid, self.y_train = np.split(self.y,
                                                 [int(len(self.y)*valid_ratio)])
            
            valid_batch_maker = BatchMaker(self.X_valid,
                                           self.y_valid,
                                           n_valid_bathces)

            validDL = DataWrapper(valid_batch_maker,
                                  function=preprocessing)
        
            train_batch_maker = BatchMaker(self.X_train,
                                           self.y_train,
                                           n_batches)
            
            trainDL = DataWrapper(train_batch_maker, 
                                  function=preprocessing)
        else:
            
            train_batch_maker = BatchMaker(self.X,
                                           self.y,
                                           n_batches)

            trainDL = DataWrapper(train_batch_maker, 
                                  function=preprocessing)
        
        if verbose:
            self.verbose_lines = ['==>','====>','======>','========>','==========>']
            self.steps_for_line_drawing = (n_batches // 5) + 1

        for epoch in range(n_epochs):
            
            for batch_num, data in enumerate(trainDL, 0):

                X, y = data
                X = np.array(X)
                X = X.reshape(X.shape[1:])
                y = np.array(y)
                y = y.reshape(y.shape[1:])
                optimizer.zero_grad()

                y_pred = self.forward(X)
                self.backward(self.loss.gradient(y_pred, y))
                self.history['train_loss'].append(self.loss.calc(y_pred, y))
                optimizer.step()

                if verbose:
                    self.display_verbose(n_batches,
                                         batch_num,
                                         n_epochs,
                                         epoch,
                                         train_mode=True)
                    # if batch_num_train == n_batches -1:
                    #     loss_ = np.sum(self.history['train_loss'][epoch*n_batches:])/n_batches
                    #     print(f'training loss: ' +
                    #           str(loss_)
                    #          )
                
            if valid_ratio:

                for batch_num, data in enumerate(validDL, 0):

                    X, y = data
                    X = np.array(X)
                    X = X.reshape(X.shape[1:])
                    y = np.array(y)
                    y = y.reshape(y.shape[1:])

                    y_pred = self.evaluate(X)
                    self.history['val_loss'].append(self.loss.calc(y_pred, y))
                
                    if verbose:
                        self.display_verbose(n_valid_bathces,
                                            batch_num,
                                            n_epochs,
                                            epoch,
                                            train_mode=False)
                    # if batch_num_valid == n_batches -1:
                    #     loss_ = np.sum(self.history['val_loss'][epoch*n_valid_bathces:])/n_valid_bathces
                    #     print('validation loss: ' + 
                    #         str(loss_)
                    #          )
    
    def evaluate(self, X_test):

        input = deepcopy(X_test)
        # attribute1 = getattr(self, self.layers[0])
        # if attribute1.bias:
        #     X_0 = np.ones(input.shape[0])
        #     X_0 = X_0[:, np.newaxis]
        #     input = np.concatenate([X_0, input], axis=1)
        #     # input = np.array(input)

        for layer in self.layers:
            attribute = getattr(self, layer)
            if hasattr(attribute, 'weights'):
                input = attribute.forward(input)
            else:
                func = NoGrad(attribute.forward)
                input = func(None, input)
        return input

    def predict(self, X_test):
        y_eval = self.evaluate(X_test)
        if y_eval.shape[1] == 1:
            y_pred = np.zeros_like(y_eval)
            y_pred[y_eval >= 0.5] = 1
            return y_pred
        else:
            y_pred = np.zeros((len(X_test),1))
            y_pred = np.argmax(y_eval,axis=1)
            return y_pred[:, np.newaxis]

    def accuracy_score(self, y_test, y_pred):
        '''
            Takes the truth and predicted values
            and calculates the accuracy score.
        '''
        return np.sum(y_pred == y_test)/len(y_test)
    
    def display_verbose(self,
                        n_batches,
                        batch_num,
                        n_epochs,
                        epoch_num,
                        train_mode):

            if train_mode:

                if n_batches == 1:
                    print('[epoch: '+str(epoch_num + 1)+'/'+str(n_epochs)+']')
                    loss_ = np.sum(self.history['train_loss'][epoch_num*n_batches:])/n_batches
                    print('training loss: ' + str(loss_))

                else:

                    if batch_num == 0:
                        print('[epoch: '+str(epoch_num + 1)+'/'+str(n_epochs)+']\ntrain:')
                        sys.stdout.write('\rbatch {one}/{two}::{three}'.format(one=batch_num +1,
                                    two=n_batches,
                                    three=self.verbose_lines[batch_num//self.steps_for_line_drawing]))
                        sys.stdout.flush()
                        sleep(.01)
                    elif batch_num == n_batches -1:
                        loss_ = np.sum(self.history['train_loss'][epoch_num*n_batches:])/n_batches
                        sys.stdout.write('\rbatch {one}/{two}::{three} training loss = {four}'.format(one=batch_num +1,
                                    two=n_batches,
                                    three=self.verbose_lines[batch_num//self.steps_for_line_drawing],
                                    four=loss_))
                        sys.stdout.flush()
                        sleep(.001)
                    else:
                        sys.stdout.write('\rbatch {one}/{two}::{three}'.format(one=batch_num +1,
                                    two=n_batches,
                                    three=self.verbose_lines[batch_num//self.steps_for_line_drawing]))
                        sys.stdout.flush()
                        sleep(.01)

            else:

                if n_batches == 1:
                    loss_ = np.sum(self.history['val_loss'][epoch_num*n_batches:])/n_batches
                    print('\nvalidation loss: ' + str(loss_) + 
                    '\n-------------------------------------')

                else:

                    if batch_num == 0:
                        print('\nvalidation:')
                    elif batch_num == n_batches -1:
                        loss_ = np.sum(self.history['val_loss'][epoch_num*n_batches:])/n_batches
                        sys.stdout.write('\rbatch {one}/{two}::{three} validation loss = {four}'.format(one=batch_num +1,
                                    two=n_batches,
                                    three=self.verbose_lines[batch_num//self.steps_for_line_drawing],
                                    four=loss_))
                        sys.stdout.flush()
                        sleep(.01)
                        print('\n-------------------------------------------')
                    else:
                        sys.stdout.write('\rbatch {one}/{two}::{three}'.format(one=batch_num +1,
                                    two=n_batches,
                                    three=self.verbose_lines[batch_num//self.steps_for_line_drawing]))
                        sys.stdout.flush()
                        sleep(.01)


    def plot_2d_boundaries(self, X_test, y_test):
        x_min, x_max = self.X_train[:,0].min() - 1, self.X_train[:,0].max() + 1
        y_min, y_max = self.X_train[:,1].min() - 1, self.X_train[:,1].max() + 1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, step=0.05),
                             np.arange(y_min, y_max, step=0.05))

        X_mesh = np.c_[xx.ravel(), yy.ravel()]
        y_pred_mesh = self.predict(X_mesh)
        y_pred_mesh = y_pred_mesh.reshape(xx.shape)

        y_pred = self.predict(X_test)
        # print('accuracy on test dataset:' +
            # f'{net.accuracy_score(y_test, y_pred)}'+
            # '\n')

        fig = go.Figure()

        fig.add_trace(
            go.Heatmap(x=np.arange(x_min,x_max,step=0.05),
                       y=np.arange(y_min,y_max,step=0.05),
                       z=y_pred_mesh,
                       colorscale='Bluered',
                       opacity=0.5,
                       showscale=False)
        )

        fig.add_trace(
            go.Scatter(x=X_test[:,0],
                       y=X_test[:,1],
                       mode='markers',
                       showlegend=False,
                       marker=dict(size=10,
                                   color=y_pred.ravel(),
                                   colorscale='Bluered',
                                   line=dict(color='black',
                                             width=1)))
        )

        fig.update_layout(
            title='Decision Boundaries & Classified Test Data Points',
            annotations= [dict(
                text='Accuracy on test dataset: ' +
                    f'{self.accuracy_score(y_test, y_pred)}'+
                     '\n',
                font=dict(
                    size=10,
                    color='black'
                    ),
                showarrow=False,
                align='center',
                x=0.5,
                y=1,
                xref='paper',
                yref='paper'
                )],
            hovermode='closest',
            xaxis_title='Feature 1',
            yaxis_title='Feature 2',
            showlegend=False
        )

        fig.show()