#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import sounddevice as sd
import numpy as np
import random
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Set random seed for reproducibility
torch.manual_seed(0)

# 定義對應的頻率和標籤
frequencies = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]

# Generate simple sine wave audio
def generate_noisy_sine_wave(freq, sample_rate=16000, duration=1, noise_level=0.05):
    t = torch.linspace(0, duration, int(sample_rate * duration), dtype=torch.float32)
    waveform = 0.5 * torch.sin(2 * torch.pi * freq * t)
    noise = noise_level * torch.randn_like(waveform)
    return waveform + noise

# Define dataset class
class SimpleAudioDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.data = []
        self.labels = []

        for _ in range(50):
            for label, freq in enumerate(frequencies):
                sine_wave = generate_noisy_sine_wave(freq)
                self.data.append(sine_wave)
                self.labels.append(label)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        waveform = self.data[index].unsqueeze(0)
        label = self.labels[index]
        if self.transform:
            waveform = self.transform(waveform)
        return waveform, label

# Define audio transformation
class AudioTransform:
    def __init__(self):
        self.mel_spectrogram = MelSpectrogram(
            sample_rate=16000,
            n_mels=32, # The numbers of filter in Mel  spectrum
            n_fft=400, # The window of FFT
            hop_length=160) # The stride(步長) between each FFT
        self.amplitude_to_db = AmplitudeToDB()

    def __call__(self, waveform):
        mel_spec = self.mel_spectrogram(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)
        return mel_spec_db

transform = AudioTransform()
dataset = SimpleAudioDataset(transform=transform)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Initialize device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define CNN model
class SimpleAudioModel(nn.Module):
    def __init__(self, sample_input):
        super(SimpleAudioModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1).to(device)
        self.pool1 = nn.MaxPool2d(2, 2).to(device)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1).to(device)
        self.pool2 = nn.MaxPool2d(2, 2).to(device)
        self.flatten = nn.Flatten().to(device)

        # Dynamically calculate flatten size using sample input
        sample_input = sample_input.to(device)
        self.flatten_size = self._get_flatten_size(sample_input)
        self.fc1 = nn.Linear(self.flatten_size, 64).to(device)
        self.fc2 = nn.Linear(64, 8).to(device)

    def _get_flatten_size(self, sample_input):
          with torch.no_grad():
              sample_input = sample_input.to(device)
              x = self.pool1(torch.relu(self.conv1(sample_input)))
              x = self.pool2(torch.relu(self.conv2(x)))
              x = self.flatten(x)
              return x.shape[1]

    def forward(self, x):
        x = x.to(device)
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model
sample_input, _ = next(iter(train_loader))
sample_input = sample_input.to(device)
model = SimpleAudioModel(sample_input).to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
def train_model(model, train_loader, criterion, optimizer):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluate the model
def evaluate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    train_model(model, train_loader, criterion, optimizer)
    accuracy = evaluate_model(model, val_loader)
    print(f"Epoch {epoch+1}, Accuracy: {accuracy:.2f}%")

# Prediction function
def predict(model, waveform):
    model.eval()
    with torch.no_grad():
        waveform = transform(waveform.unsqueeze(0)).to(device)
        output = model(waveform.unsqueeze(0))
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Function to play audio
def play_audio(waveform, sample_rate=16000):
    # Ensure audio is on CPU and convert to NumPy
    waveform_np = waveform.numpy()
    sd.play(waveform_np, sample_rate)
    sd.wait()

#%%

# Test model prediction and audio playback
for freq in frequencies:
    test_waveform = generate_noisy_sine_wave(freq)
    predicted_label = predict(model, test_waveform)
    print(f"Predicted label for {freq} Hz: {predicted_label}")
    play_audio(test_waveform)

