import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from jetnet.datasets import JetNet
from jetnet.datasets.normalisations import FeaturewiseLinear
from sklearn.preprocessing import OneHotEncoder


# Define function for one-hot encoding
def OneHotEncodeType(x):
    enc = OneHotEncoder(categories=[[0, 2, 3]]) 
    type_encoded = enc.fit_transform(x[..., 0].reshape(-1, 1)).toarray()
    other_features = x[..., 1:].reshape(-1, 3)
    return np.concatenate((type_encoded, other_features), axis=-1).reshape(*x.shape[:-1], -1)

# Dataset preparation parameters
data_args = {
    "jet_type": ["g", "t", "w"],  # gluon, top quark, and W boson jets
    "data_dir": "datasets/jetnet",  # data directory
    "particle_features": ["etarel", "phirel", "ptrel", "mask"],
    "num_particles": 10,  # retain only 10 highest pT particles
    "jet_features": ["type", "pt", "eta", "mass"],
    "particle_normalisation": FeaturewiseLinear(
        normal=True, normalise_features=[True, True, True, False]
    ),
    "jet_transform": OneHotEncodeType,  # one-hot encode jet type
    "download": True,  # download dataset if not already available
}
# Load datasets
jets_train = JetNet(**data_args, split="train")
jets_valid = JetNet(**data_args, split="valid")

# Define DataLoader parameters
batch_size = 64
shuffle = True

# Create DataLoaders
train_loader = DataLoader(jets_train, batch_size=batch_size, shuffle=shuffle)
valid_loader = DataLoader(jets_valid, batch_size=batch_size, shuffle=shuffle)

print("Train and validation DataLoaders are ready.")

# Define a minimal Transformer model
class MiniTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, hidden_dim, output_dim):
        super(MiniTransformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.transformer(x)  # Apply transformer encoder
        x = x.mean(dim=1)  # Aggregate over the sequence (mean pooling)
        x = self.fc(x)  # Final output layer
        return x


# Model parameters
input_dim = 4  # Particle features
num_heads = 2
num_layers = 2
hidden_dim = 128
output_dim = 3  # Multi-class classification for jet types

# Initialize model, loss, and optimizer
model = MiniTransformer(input_dim, num_heads, num_layers, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Training loop for one epoch
model.train()
for batch_idx, (particle_features, jet_features) in enumerate(train_loader):
    # Move data to the device
    particle_features = particle_features.to(device)  # [64, 30, 4]
    jet_labels = jet_features[:, 0:3].to(device)  # Class labels (0, 2, 3)

    # Forward pass
    outputs = model(particle_features)  # [64, 3]
    
    # Calculate loss
    loss = criterion(outputs, jet_labels)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print training status
    if batch_idx % 10 == 0:
        print(f"Batch {batch_idx + 1}, Loss: {loss.item():.4f}")

print("Training for one epoch completed.")
