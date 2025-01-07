import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import jetnet
from jetnet.datasets import JetNet
from jetnet.datasets.normalisations import FeaturewiseLinear
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# function to one hot encode the jet type and leave the rest of the features as is
def OneHotEncodeType(x: np.ndarray):
    enc = OneHotEncoder(categories=[[0., 2., 3.]])
    type_encoded = enc.fit_transform(x[..., 0].reshape(-1, 1)).toarray()
    other_features = x[..., 1:].reshape(-1, 3)
    return np.concatenate((type_encoded, other_features), axis=-1).reshape(*x.shape[:-1], -1)


data_args = {
    "jet_type": ["g", "t", "w"],  # gluon and top quark jets
    "data_dir": "./datasets/jetnet",
    # these are the default particle features, written here to be explicit
    "particle_features": ["etarel", "phirel", "ptrel", "mask"],
    # "num_particles": 10,  # we retain only the 10 highest pT particles for this demo
    "jet_features": ["type", "pt", "eta", "mass"],
    # we don't want to normalise the 'mask' feature so we set that to False
    "particle_normalisation": FeaturewiseLinear(
        normal=True, normalise_features=[True, True, True, False]
    ),
    # pass our function as a transform to be applied to the jet features
    "jet_transform": OneHotEncodeType,
    "download": True,
}

jets_train = JetNet(**data_args, split="train")
jets_valid = JetNet(**data_args, split="valid")

x_train = jets_train.particle_data
y_train = jets_train.jet_data[:, 0]
x_valid = jets_valid.particle_data
y_valid = jets_valid.jet_data[:, 0]


x_train_flat = torch.from_numpy(x_train.reshape(x_train.shape[0], -1))
x_valid_flat = torch.from_numpy(x_valid.reshape(x_valid.shape[0], -1))

y_train = y_train.reshape(-1,1)
y_valid = y_valid.reshape(-1,1)

enc = OneHotEncoder(sparse_output=False)
y_train_encoded = enc.fit_transform(y_train)
y_valid_encoded = enc.fit_transform(y_valid)


y_train_indices = torch.from_numpy(np.argmax(y_train_encoded, axis=1))
y_valid_indices = torch.from_numpy(np.argmax(y_valid_encoded, axis = 1))

# Define MLP Model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Model Parameters
input_size = 120  # Flattened feature size
hidden_size = 64    # Number of hidden units
model = MLP(input_size, hidden_size, num_classes=3)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()  # Automatically handles one-hot or class indices
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 800
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(x_train_flat)
    loss = criterion(outputs, y_train_indices)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Predict
with torch.no_grad():
    predictions = torch.argmax(model(x_valid_flat), dim=1)
    
accuracy = accuracy_score(y_valid_indices, predictions)
