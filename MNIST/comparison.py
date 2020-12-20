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



class DataWrapper():
    def __init__(self, DL):
        self.DL = DL

    def __len__(self):
        return len(self.DL)

    def __iter__(self):
        batches = iter(self.DL)
        for b in batches:
            yield b

trainDL = DataWrapper(trainDL)
validDL = DataWrapper(validDL)

# %%
class Model_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=784,
                             out_features=10)
        self.activation1 = nn.ReLU()
        self.activation2 = nn.Softmax()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation2(x)
        return x

class Model_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=784,
                             out_features=32)
        self.fc2 = nn.Linear(in_features=32,
                             out_features=10)
        self.activation1 = nn.ReLU()
        self.activation2 = nn.Softmax()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation1(x)
        x = self.fc2(x)
        x = self.activation2(x)
        return x

model_1 = Model_1()
model_1.to(device)
print(model_1)
model_2 = Model_2()
model_2.to(device)
print(model_2)
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
for layer in model_1.children():
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()

for layer in model_2.children():
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()

criterion = nn.CrossEntropyLoss()
optimizer_1 = optim.SGD(model_1.parameters(), lr=0.001, )
optimizer_2 = optim.SGD(model_2.parameters(), lr=0.001, )
num_epochs = 100
fit(trainDL, validDL, model_1, criterion, optimizer_1, num_epochs)
fit(trainDL, validDL, model_2, criterion, optimizer_2, num_epochs)
# %%
from sklearn.metrics import classification_report, confusion_matrix

X_test = xValid.clone()
X_test = X_test.to(device)

with torch.no_grad():
    y_pred_1 = model_1(X_test)
    y_pred_1 = torch.argmax(y_pred_1, axis=1)
y_pred_1 = y_pred_1.cpu()
y_pred_1 = y_pred_1.numpy()
print(classification_report(y_pred_1, yValid))

with torch.no_grad():
    y_pred_2 = model_2(X_test)
    y_pred_2 = torch.argmax(y_pred_2, axis=1)
y_pred_2 = y_pred_2.cpu()
y_pred_2 = y_pred_2.numpy()
print(classification_report(y_pred_2, yValid))
# %%
print(confusion_matrix(y_pred_1, yValid))
print('\n')
print(confusion_matrix(y_pred_2, yValid))
# %%
