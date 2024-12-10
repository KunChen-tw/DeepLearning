import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# Set the hyperparameters
latent_dim = 1
data_dim = 2
batch_size = 32
epochs = 100000
lr = 0.001

# Generate real-world data: (x, y) pairs
def real_data_sampler(num_samples):
    x = np.linspace(-1, 1, num_samples)
    y = x**2
    data = np.vstack((x, y)).T
    return data

# Generator model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, data_dim)
        )
    def forward(self, z):
        return self.model(z)

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(data_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, data):
        return self.model(data)

generator = Generator()
discriminator = Discriminator()

# Loss function
adversarial_loss = nn.BCELoss()

# Optimizer
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# Train the GAN model
for epoch in range(epochs):
    # Training discriminator    
    optimizer_D.zero_grad()  
    # Real data
    real_data = torch.Tensor(real_data_sampler(batch_size))
    valid = torch.ones(batch_size, 1)
    fake = torch.zeros(batch_size, 1)
    real_loss = adversarial_loss(discriminator(real_data), valid)
    # Fake data    
    z = torch.randn(batch_size, latent_dim)
    generated_data = generator(z)
    fake_loss = adversarial_loss(discriminator(generated_data), fake)
    d_loss = (real_loss + fake_loss) / 2
    d_loss.backward()
    optimizer_D.step() # Update the parameters of discriminator
    
    # Training generator    
    optimizer_G.zero_grad()
    z = torch.randn(batch_size, latent_dim)
    generated_data = generator(z)
    g_loss = adversarial_loss(discriminator(generated_data), valid)   
    g_loss.backward()
    optimizer_G.step() # Update the parameters of generator

    # Outputs training progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{epochs} [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")


# Plot the resulting data points
z = torch.randn(100, latent_dim)
generated_data = generator(z).detach().numpy()
real_data = real_data_sampler(100)

plt.scatter(real_data[:, 0], real_data[:, 1], color='red', label='Real data')
plt.scatter(generated_data[:, 0], generated_data[:, 1], color='blue', label='Generated data')
plt.legend()
plt.show()
