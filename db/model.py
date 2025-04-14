# model.py

import torch
import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    def __init__(self, latent_dim=100, out_channels=4, height=128, width=500):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.height = height
        self.width = width

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * 8 * 16),
            nn.ReLU()
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 32 -> 64
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, out_channels, kernel_size=(2, 4), stride=(2, 2), padding=(0, 1)),  # 64x32 -> 128x64
            nn.Tanh()
        )

        self.upsample = nn.Upsample(size=(height, width), mode='bilinear', align_corners=True)

    def forward(self, z):
        out = self.fc(z)
        out = out.view(z.size(0), 512, 8, 16)
        out = self.deconv(out)
        out = self.upsample(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        c, h, w = input_shape
        self.model = nn.Sequential(
            nn.Conv2d(c, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Flatten(),
            nn.Linear((h // 4) * (w // 4) * 128, 1)  # 输出不是 sigmoid（LSGAN）
        )

    def forward(self, x):
        return self.model(x)
