import numpy as np


class Activation(object):

    def __new__(cls, name, cross_entropy_loss=True):
    # def __new__(cls, name):
        assert name in ['sigmoid', 'relu', 'softmax'], 'Invalid function name'
        if name == 'sigmoid':
            return Sigmoid()
        elif name == 'relu':
            return Relu()
        elif name == 'softmax':
            return Softmax(cross_entropy_loss)

# class Sigmoid:

#     @staticmethod
#     def forward(ctx, input, requires_grad=True):
#         '''
#             Takes an array as its argument
#             and returns calculated sigmoid 
#             function of the input.
#         '''
#         if requires_grad:
#             # SAVE ACTIVATION INPUT FOR BACKWARD PASS
#             ctx.activation_inputs.append(input)  
#             output = 1./(1. + np.exp(-input))
#             ctx.inputs.append(output)  # FOR GRADEINT OF PARAMETERS
#         else:
#             output = 1./(1. + np.exp(-input))
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         '''
#             Takes an array which is the ouput 
#             of a sigmoid function as its argument
#             and returns the derivative of
#             sigmoid function.
#         '''
#         input = ctx.activation_inputs.pop()
#         output = 1./(1. + np.exp(-input))
#         output = output * (1. - output)
#         grad_sigmoid = grad_output.copy()
#         return output * grad_sigmoid 


# class Softmax:

#     def __init__(self, cross_entropy_loss):
#         self.cross_entropy_loss = cross_entropy_loss

#     @staticmethod
#     def forward(ctx, input, requires_grad=True):
#         '''
#             Takes an array as its argument
#             and returns calculated sigmoid 
#             function of the input.
#         '''
#         if requires_grad:
#             ctx.activation_inputs.append(input)
#             output = np.exp(input)/np.reshape(np.sum(np.exp(input),axis=1), (-1,1))
#             ctx.inputs.append(output)
#         else:
#             output = np.exp(input)/np.reshape(np.sum(np.exp(input),axis=1), (-1,1))
#         return output

#     def backward(self, ctx, grad_output):
#         '''
#             Takes an array which is the ouput 
#             of a sigmoid function as its argument
#             and returns the derivative of
#             sigmoid function.
#         '''
#         if self.cross_entropy_loss:
#             output = grad_output.copy()
#             ctx.activation_inputs.pop()
#         else:
#             input = ctx.activation_inputs.pop()
#             # softmax = np.exp(input)/np.reshape(np.sum(np.exp(input),axis=1), (-1,1))
#             softmax = np.exp(input)/np.reshape(np.sum(np.exp(input),axis=1), (-1,1) + 1e-5)
#             grad_ = grad_output.copy()
#             output = np.zeros_like(grad_)
#             for i in range(len(softmax)):
#                 softmax_ = softmax[i]
#                 softmax_ = np.reshape(softmax_,(1,-1))
#                 softmax_derivative = softmax_ * np.identity(softmax_.size) -\
#                                         np.matmul(softmax_.T, softmax_)
#                 output[i] = np.matmul(grad_[i], softmax_derivative)
#         return output
    
# class Relu:

#     @staticmethod
#     def forward(ctx, input, requires_grad=True):
#         '''
#             Takes an array as its argument
#             and returns calculated sigmoid 
#             function of the input.
#         '''
#         if requires_grad:
#             ctx.activation_inputs.append(input)
#             output = np.clip(input, 0., np.max(input))
#             ctx.inputs.append(output)
#         else:
#             output = np.clip(input, 0., np.max(input))
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         '''
#             Takes an array which is the ouput 
#             of a sigmoid function as its argument
#             and returns the derivative of
#             sigmoid function.
#         '''
#         input = ctx.activation_inputs.pop()
#         grad_new = grad_output.copy()
#         grad_new[input < 0. ] = 0.
#         return grad_new




class Sigmoid:

    @staticmethod
    def forward(ctx, input, requires_grad=True):
        '''
            Takes an array as its argument
            and returns calculated sigmoid 
            function of the input.
        '''
        if requires_grad:
            # SAVE ACTIVATION INPUT FOR BACKWARD PASS
            ctx.activation_inputs.append(input)  
            output = 1./(1. + np.exp(-input))
            ctx.inputs.append(output)  # FOR GRADEINT OF PARAMETERS
        else:
            output = 1./(1. + np.exp(-input))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        '''
            Takes an array which is the ouput 
            of a sigmoid function as its argument
            and returns the derivative of
            sigmoid function.
        '''
        input = ctx.activation_inputs.pop()
        output = 1./(1. + np.exp(-input))
        output = output * (1. - output)
        grad_sigmoid = grad_output.copy()
        grad = output * grad_sigmoid 
        # grad = np.clip(grad, 
        #                 np.min(grad) / np.linalg.norm(grad), 
        #                 np.max(grad) / np.linalg.norm(grad)))
        # return grad * 1./ np.linalg.norm(grad)
        return np.clip(grad, -1., 1.)


class Softmax:

    def __init__(self, cross_entropy_loss):
        self.cross_entropy_loss = cross_entropy_loss

    @staticmethod
    def forward(ctx, input, requires_grad=True):
        '''
            Takes an array as its argument
            and returns calculated sigmoid 
            function of the input.
        '''
        if requires_grad:
            ctx.activation_inputs.append(input)
            output = np.exp(input)/np.reshape(np.sum(np.exp(input),axis=1), (-1,1))
            ctx.inputs.append(output)
        else:
            output = np.exp(input)/np.reshape(np.sum(np.exp(input),axis=1), (-1,1))
        return output

    def backward(self, ctx, grad_output):
        '''
            Takes an array which is the ouput 
            of a sigmoid function as its argument
            and returns the derivative of
            sigmoid function.
        '''
        if self.cross_entropy_loss:
            output = grad_output.copy()
            ctx.activation_inputs.pop()
        else:
            input = ctx.activation_inputs.pop()
            softmax = np.exp(input)/(np.reshape(np.sum(np.exp(input),axis=1), (-1,1)) + 1e-5)
            # softmax = np.exp(input)/(np.reshape(np.sum(np.exp(input),axis=1), (-1,1)))
            grad_ = grad_output.copy()
            output = np.zeros_like(grad_)
            for i in range(len(softmax)):
                softmax_ = softmax[i]
                softmax_ = np.reshape(softmax_,(1,-1))
                softmax_derivative = softmax_ * np.identity(softmax_.size) -\
                                        np.matmul(softmax_.T, softmax_)
                output[i] = np.matmul(grad_[i], softmax_derivative)
        grad = output
        # grad = np.clip(grad, 
        #                 np.min(grad) / np.linalg.norm(grad), 
        #                 np.max(grad) / np.linalg.norm(grad)))
        # return grad * 1./ np.linalg.norm(grad)
        return np.clip(grad, -1., 1.)
    
class Relu:

    @staticmethod
    def forward(ctx, input, requires_grad=True):
        '''
            Takes an array as its argument
            and returns calculated sigmoid 
            function of the input.
        '''
        if requires_grad:
            ctx.activation_inputs.append(input)
            output = np.clip(input, 0., np.max(input))
            ctx.inputs.append(output)
        else:
            output = np.clip(input, 0., np.max(input))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        '''
            Takes an array which is the ouput 
            of a sigmoid function as its argument
            and returns the derivative of
            sigmoid function.
        '''
        input = ctx.activation_inputs.pop()
        grad_new = grad_output.copy()
        grad_new[input < 0. ] = 0.
        grad = grad_new.copy()
        # grad = np.clip(grad, 
        #                 np.min(grad) / np.linalg.norm(grad), 
        #                 np.max(grad) / np.linalg.norm(grad)))
        # return grad * 1./ np.linalg.norm(grad)
        return np.clip(grad, 0., 1.)