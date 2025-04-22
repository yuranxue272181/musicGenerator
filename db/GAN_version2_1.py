import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pretty_midi

# === 数据集预处理 ===
def find_all_midi_files(root_dir):
    midi_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(('.mid', '.midi')):
                midi_files.append(os.path.join(dirpath, filename))
    return midi_files

def midi_to_multi_piano_roll(midi_file, fs=100, max_tracks=4, fixed_length=500):
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file)
        tracks = []
        for instrument in midi_data.instruments[:max_tracks]:
            roll = instrument.get_piano_roll(fs=fs) / 127.0
            tracks.append(roll)
        while len(tracks) < max_tracks:
            tracks.append(np.zeros_like(tracks[0]))
        processed = []
        for roll in tracks:
            roll = roll[:, :fixed_length]
            if roll.shape[1] < fixed_length:
                pad_width = fixed_length - roll.shape[1]
                roll = np.pad(roll, ((0, 0), (0, pad_width)))
            processed.append(roll)
        return np.stack(processed, axis=0)
    except Exception as e:
        print(f"Error processing {midi_file}: {e}")
        return None

class MidiDatasetMulti(Dataset):
    def __init__(self, midi_dir, fs=100, fixed_length=500, max_tracks=4):
        self.files = find_all_midi_files(midi_dir)
        self.fs = fs
        self.fixed_length = fixed_length
        self.max_tracks = max_tracks
        self.data = [midi_to_multi_piano_roll(f, fs, max_tracks, fixed_length) for f in self.files]
        self.data = [d for d in self.data if d is not None]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)

# === CNN Generator ===
class CNNGenerator(nn.Module):
    def __init__(self, latent_dim=100, output_shape=(4, 128, 500)):
        super(CNNGenerator, self).__init__()
        self.channels, self.pitch, self.time = output_shape
        self.init_shape = (64, self.pitch // 4, self.time // 8)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, int(np.prod(self.init_shape))),
            nn.ReLU()
        )
        self.conv_blocks = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=(1, 2)),
            nn.Conv2d(32, self.channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(z.size(0), *self.init_shape)
        return self.conv_blocks(out)

# === Discriminator (仍为 MLP) ===
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(input_shape), 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

# === 训练函数 ===
def train_gan(midi_dir, epochs=50, batch_size=16, latent_dim=100, output_shape=(4, 128, 500)):
    dataset = MidiDatasetMulti(midi_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    generator = CNNGenerator(latent_dim, output_shape)
    discriminator = Discriminator(output_shape)

    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        for i, real in enumerate(dataloader):
            bs = real.size(0)
            valid = torch.ones(bs, 1) * 0.9
            fake = torch.zeros(bs, 1)

            # Train D
            optimizer_D.zero_grad()
            real_loss = criterion(discriminator(real), valid)
            z = torch.randn(bs, latent_dim)
            gen = generator(z)
            fake_loss = criterion(discriminator(gen.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # Train G
            optimizer_G.zero_grad()
            gen = generator(z)
            g_loss = criterion(discriminator(gen), valid)
            g_loss.backward()
            optimizer_G.step()

            if i % 10 == 0:
                print(f"[Epoch {epoch+1}/{epochs}] [Batch {i}] D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    midi_dir = os.path.join(base_dir, "fixed_midi")
    train_gan(midi_dir)
