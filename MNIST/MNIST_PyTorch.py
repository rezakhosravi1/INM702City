# %% 
from pathlib2 import Path
import pickle
import gzip
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
# %%
parentPath = Path('data')
path = parentPath / 'mnist'
fileName = 'mnist.pkl.gz'

with gzip.open((path / fileName).as_posix(), 'rb') as f:
    ((xTrain, yTrain), (xValid, yValid), _) = pickle.load(f,
                                                    encoding='latin-1')
# %%
# converting the data to torch tensor
device = torch.device('cuda')
xTrain, yTrain, xValid, yValid = map(torch.tensor,
                                    (xTrain, yTrain, xValid, yValid))

trainDS = TensorDataset(xTrain, yTrain)
validDS = TensorDataset(xValid, yValid)

batch_size = 64
trainDL = DataLoader(trainDS, batch_size=batch_size, shuffle=True)
validDL = DataLoader(validDS, batch_size=batch_size * 2)

# wrapping the data
def Preprocessing(x, y):
    return x.view(-1, 1, 28, 28), y

class DataWrapper():
    def __init__(self, DL, func):
        self.DL = DL
        self.func = func

    def __len__(self):
        return len(self.DL)

    def __iter__(self):
        batches = iter(self.DL)
        for b in batches:
            yield (self.func(*b))

trainDL = DataWrapper(trainDL, Preprocessing)
validDL = DataWrapper(validDL, Preprocessing)

# %%

# model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                                out_channels=6,
                                kernel_size=5,
                                stride=1,
                                padding=1)
        self.conv2_1 = nn.Conv2d(in_channels=6,
                                out_channels=16,
                                kernel_size=5,
                                stride=2,
                                padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=6,
                                out_channels=16,
                                kernel_size=5,
                                stride=2,
                                padding=1)
        self.conv3 = nn.Conv2d(in_channels=16,
                                out_channels=120,
                                kernel_size=5,
                                stride=2,
                                padding=1)
        self.fc1 = nn.Linear(in_features=120,
                             out_features=84)
        self.fc2 = nn.Linear(in_features=84,
                             out_features=10)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.activation1 = nn.ReLU()
        self.activation2 = nn.Softmax()

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.max_pool(x)

        x_1 = self.conv2_1(x)
        x_1 = self.activation1(x_1)
        x_1 = self.max_pool(x_1)
        
        x_2 = self.conv2_2(x)
        x_2 = self.activation1(x_2)
        x_2 = self.max_pool(x_2)
        
        x_3 = x_1 + x_2
        
        x_3 = self.conv3(x_3)
        x_3 = self.activation1(x_3)

        x_4 = x_3.view(x.size(0), -1)

        x_4 = self.fc1(x_4)
        x_4 = self.activation1(x_4)

        x_4 = self.fc2(x_4)
        x_4 = self.activation2(x_4)

        return x_4

model = Model()
model.to(device)
# %%
# training the model
def fit(trainDL, validDL, model, criterion, optimizer, num_epochs):
    for epoch in range(1, num_epochs + 1):
        trainLoss = 0.
        validLoss = 0.
        trainLength = 0
        validLength = 0

        model.train()
        for cnt, data in enumerate(trainDL, 0):
            xb, yb = data[0].to(device), data[1].to(device)
            ypred = model(xb)
            trainLength += len(xb)

            loss = criterion(ypred, yb)
            trainLoss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch: {a} train loss: {b}'.format(
                                a=epoch,
                                b=trainLoss/trainLength))

        model.eval()
        with torch.no_grad():
            for cnt, data in enumerate(validDL, 0):
                xb, yb = data[0].to(device), data[1].to(device)
                ypred = model(xb)
                validLength += len(xb)

                loss = criterion(ypred, yb)
                validLoss += loss

        print('epoch: {a} validation loss: {b}'.format(
                            a=epoch,
                            b=validLoss/validLength))
# %%
for layer in model.children():
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001,)
num_epochs = 100
fit(trainDL, validDL, model, criterion, optimizer, num_epochs)
# %%
from sklearn.metrics import classification_report, confusion_matrix
X_test = xValid.clone()
X_test = X_test.view(-1,1,28,28)
X_test = X_test.to(device)
# %%
with torch.no_grad():
    y_pred = model(X_test)
    y_pred = torch.argmax(y_pred, axis=1)
y_pred = y_pred.cpu()
y_pred = y_pred.numpy()
print(classification_report(y_pred, yValid))
# %%
print(confusion_matrix(y_pred, yValid))
# %%
# %%
# THIS RETURNS THE ARCHITECTURE
import tensorflow as tf 
tfk = tf.keras
tfkl = tf.keras.layers

# JUST FOR PLOTTING REASONS THE OUTPUT SHAPES ARE DIFFERENT
input_image = tfk.Input(shape=(28,28,1),name='input')
output = tfkl.Conv2D(6, 5, 1, activation='relu', name='conv1', padding='valid')(input_image)
output = tfkl.MaxPool2D(2,2)(output)
output_1 = tfkl.Conv2D(16, 5, 1, activation='relu', name='conv2_1', padding='valid')(output)
output_1 = tfkl.MaxPool2D(2,2)(output_1)
output_2 = tfkl.Conv2D(16, 5, 1, activation='relu', name='conv2_2', padding='valid')(output)
output_2 = tfkl.MaxPool2D(2,2)(output_2)
output_add = tfkl.Add()([output_1, output_2])
output_3 = tfkl.Conv2D(120, 5, 1, activation='relu', name='conv3', padding='same')(output_add)
output_flat = tfkl.Flatten()(output_3)
output_fc_1 = tfkl.Dense(84, activation='relu', name='fc_1')(output_flat)
final_output = tfkl.Dense(10, activation='softmax', name='output')(output_fc_1) 
model = tfk.Model(inputs=input_image, outputs=final_output, name='LeNet5')

# %%
model.summary()
# %%
tfk.utils.plot_model(model, 'LeNet5withPadding.png')
# %%
