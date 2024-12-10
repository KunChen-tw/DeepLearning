import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Set parameters
batch_size = 64
learning_rate = 0.0002
num_epochs = 100
latent_dim = 100

# Dataset (using MNIST)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
data_loader = DataLoader(mnist_data, batch_size=batch_size, shuffle=True)

# Define generator network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x).view(-1, 1, 28, 28)


# Define discriminator network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x.view(-1, 28 * 28))

# Initialize generator and discriminator, and move them to the device
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Set loss function and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

# Function to display generated images (only one window will be created)
def show_generated_images(epoch, generator, num_images=4):
    generator.eval()  # Set generator to evaluation mode
    with torch.no_grad():  # No need for gradient computation
        # Generate random noise and images
        z = torch.randn(num_images, latent_dim).to(device)
        generated_images = generator(z).cpu()  # Generate images and move to CPU for display

        # Clear the current figure and update in-place
        for ax in axes.flatten():
            ax.clear()  # Clear each subplot

        for i, ax in enumerate(axes.flatten()):
            ax.imshow(generated_images[i].squeeze(), cmap='gray')
            ax.axis('off')

        plt.tight_layout()
        plt.draw()  # Redraw the updated figure
        plt.pause(0.1)  # Pause to allow for screen update

# Set up the plot only once (before training starts)
plt.ion()  # Turn on interactive mode for real-time plotting
fig, axes = plt.subplots(2, 2, figsize=(5, 5))  # Only one figure and axes object
plt.tight_layout()

# Start training
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        batch_size = images.size(0) 
        images = images.to(device)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        # Train discriminator
        optimizer_D.zero_grad()
        outputs = discriminator(images)
        d_loss_real = criterion(outputs, real_labels)
        
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = generator(z)
        outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()
        # Train generator
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = generator(z)
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)
        
        g_loss.backward()
        optimizer_G.step()  
    # Generate some images at the end of each epoch
    show_generated_images(epoch, generator)
    print(f'Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')

# Turn off interactive mode at the end of training
plt.ioff()
plt.show()
