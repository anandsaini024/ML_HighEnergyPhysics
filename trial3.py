import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from jetnet.datasets import JetNet
from jetnet.datasets.normalisations import FeaturewiseLinear
from sklearn.preprocessing import OneHotEncoder

# Function to one-hot encode the jet type and leave the rest of the features as is
def OneHotEncodeType(x: np.ndarray):
    enc = OneHotEncoder(categories=[[0.0, 2.0, 3.0]])
    type_encoded = enc.fit_transform(x[..., 0].reshape(-1, 1)).toarray()
    other_features = x[..., 1:].reshape(-1, 3)
    return np.concatenate((type_encoded, other_features), axis=-1).reshape(*x.shape[:-1], -1)

# Data arguments
data_args = {
    "jet_type": ["g", "t", "w"],
    "data_dir": "./datasets/jetnet",
    "particle_features": ["etarel", "phirel", "ptrel", "mask"],
    "jet_features": ["type", "pt", "eta", "mass"],
    "particle_normalisation": FeaturewiseLinear(
        normal=True, normalise_features=[True, True, True, False]
    ),
    "jet_transform": OneHotEncodeType,
    "download": True,
}

# Load datasets
jets_train = JetNet(**data_args, split="train")
jets_valid = JetNet(**data_args, split="valid")

train_loader = DataLoader(jets_train, batch_size=64, shuffle=True)
val_loader = DataLoader(jets_valid, batch_size=64, shuffle=False)

# Transformer model for jet tagging
class MiniTransformer(nn.Module):
    def __init__(self, num_classes=3, d_model=128, nhead=4, num_layers=2):
        super(MiniTransformer, self).__init__()
        self.embedding = nn.Linear(4, d_model)  # Input features: 4 (particle features)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, batch_first=True
        )
        self.fc = nn.Linear(d_model, num_classes)  # Output: num_classes (one-hot encoded jet type)

    def forward(self, x):
        x = self.embedding(x)  # Shape: [batch_size, num_particles, d_model]
        x = self.transformer(x, x)  # Shape: [batch_size, num_particles, d_model]
        x = x.mean(dim=1)  # Pooling over particles
        x = self.fc(x)  # Shape: [batch_size, num_classes]
        return x

# Model, loss, and optimizer
model = MiniTransformer()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def slice_jet_features(jet_features):
    return jet_features[:, :3]  # Extract first 3 columns (one-hot encoded jet type)

# Training loop
def train_model(model, train_loader, val_loader, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for particle_features, jet_features in train_loader:
            particle_features = particle_features.to(device)
            jet_features = slice_jet_features(jet_features).to(device)
            
            # Convert one-hot encoded target to class indices
            target = torch.argmax(jet_features, dim=1)

            # Forward pass
            outputs = model(particle_features)
            loss = criterion(outputs, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for particle_features, jet_features in val_loader:
                particle_features = particle_features.to(device)
                jet_features = slice_jet_features(jet_features).to(device)
                
                target = torch.argmax(jet_features, dim=1)

                outputs = model(particle_features)
                loss = criterion(outputs, target)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        print(
            f"Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {100 * correct / total:.2f}%"
        )

# Train the model
train_model(model, train_loader, val_loader, num_epochs=10)
