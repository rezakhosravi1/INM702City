# %% 
import numpy as np
from pathlib2 import Path
import pickle
import gzip

# %%
parentPath = Path('data')
path = parentPath / 'mnist'
fileName = 'mnist.pkl.gz'

with gzip.open((path / fileName).as_posix(), 'rb') as f:
    ((X_train, y_train), (X_test, y_test), _) = pickle.load(f,
                                                    encoding='latin-1')

y_train = y_train[:, np.newaxis]
y_test = y_test[:, np.newaxis]
# %%
X_train.shape
# %%
from MLP import nn
class Net(nn.Model):

    def __init__(self, name):
        super().__init__(name)
        self.fc1 = nn.layers.Linear(784, 10, True)
        self.activation1 = nn.activations.Activation('relu')
        self.activation2 = nn.activations.Activation('softmax')

net = Net('my_model')
net.compile(['fc1', 'activation2', ])
optimizer = nn.optimizers.Optimizer('sgd', net.parameters(), .0001)
loss = nn.losses.Loss('softmax_cce')
# %%
y_train_encoded = np.zeros((y_train.shape[0],np.max(y_train)+1))
for i in range(np.max(y_train)+1):
    y_train_encoded[y_train.ravel()==i,i] = 1
# for x,y in     
net.fit(X_train[0:50000], y_train_encoded[0:50000], optimizer, loss, 
        n_epochs=100, n_batches=780, 
        valid_ratio=0.1, n_valid_bathces=1,
        verbose=True)

# %%
y_pred = net.predict(X_test)
net.accuracy_score(y_test,y_pred)

# %%
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_pred, y_test))
print(confusion_matrix(y_pred, y_test))
# %%
