import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import jetnet
from jetnet.datasets import JetNet
from jetnet.datasets.normalisations import FeaturewiseLinear
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader

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

train_loader = DataLoader(jets_train, batch_size=64, shuffle=True)
val_loader = DataLoader(jets_valid, batch_size=64, shuffle=True)

for i, (particle_batch, jet_batch) in enumerate(train_loader):
    print(f"Batch {i+1}:")
    print(f"Particle Batch Shape: {particle_batch.shape}")
    print(f"Jet Batch Shape: {jet_batch.shape}")
    break 


class HEPTransformer(nn.Module):
    def __init__(self, particle_dim, jet_dim, embed_dim, num_heads, ff_dim, num_layers, output_dim):
        super(HEPTransformer, self).__init__()
        
        # Particle embedding
        self.particle_embedding = nn.Linear(particle_dim, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(30, embed_dim))  # Max particles = 30

        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim),
            num_layers=num_layers
        )
        
        # Jet feature processing
        self.jet_processor = nn.Linear(jet_dim, embed_dim)

        # Final classification head
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, output_dim)
        )

    def forward(self, particle_features, jet_features):
        # Embed particle features and add positional encoding
        batch_size, num_particles, _ = particle_features.shape
        particle_features = self.particle_embedding(particle_features)  # [B, 30, embed_dim]
        particle_features += self.positional_encoding[:num_particles]  # Add positional encoding

        # Process particle data with transformer
        particle_features = particle_features.transpose(0, 1)  # [30, B, embed_dim]
        encoded_particles = self.transformer(particle_features).mean(dim=0)  # Mean pool [B, embed_dim]

        # Process jet features
        processed_jet = self.jet_processor(jet_features)  # [B, embed_dim]

        # Combine particle and jet representations
        combined = encoded_particles + processed_jet  # Element-wise addition

        # Final prediction
        output = self.output_layer(combined)  # [B, output_dim]
        return output

# Hyperparameters
particle_dim = 4
jet_dim = 6
embed_dim = 64
num_heads = 4
ff_dim = 128
num_layers = 2
output_dim = 10  # Example: 10-class classification

# Initialize model, loss, and optimizer
model = HEPTransformer(particle_dim, jet_dim, embed_dim, num_heads, ff_dim, num_layers, output_dim)
criterion = nn.CrossEntropyLoss()  # Classification loss
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(10):  # Train for 10 epochs
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        particle_features, jet_features, targets = batch  # Assuming train_loader returns these
        particle_features = particle_features.to(device)  # [64, 30, 4]
        jet_features = jet_features.to(device)  # [64, 6]
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(particle_features, jet_features)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")