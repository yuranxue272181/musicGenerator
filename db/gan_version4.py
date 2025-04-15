# train_gan_music.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from model import Generator, Discriminator
from utils import reward_from_rhythm, reward_from_density
from pretty_midi import PrettyMIDI

import pretty_midi
import glob


# -----------------------------
# 数据加载（与之前一致）
# -----------------------------
def find_all_midi_files(root_dir):
    midi_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(('.mid', '.midi')):
                midi_files.append(os.path.join(dirpath, filename))
    return midi_files


def midi_to_multi_piano_roll(midi_file, fs=100, max_tracks=4, fixed_length=1000):
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file)
        tracks = []
        for instrument in midi_data.instruments:
            roll = instrument.get_piano_roll(fs=fs)
            roll = roll / 127.0
            tracks.append(roll)
            if len(tracks) >= max_tracks:
                break
        if len(tracks) < max_tracks:
            max_time = max((r.shape[1] for r in tracks), default=0)
            for _ in range(max_tracks - len(tracks)):
                tracks.append(np.zeros((128, max_time)))
        target_length = fixed_length
        processed = []
        for roll in tracks:
            if roll.shape[1] < target_length:
                pad_width = target_length - roll.shape[1]
                roll = np.pad(roll, ((0, 0), (0, pad_width)), mode='constant')
            else:
                roll = roll[:, :target_length]
            processed.append(roll)
        multi_roll = np.stack(processed, axis=0)
        return multi_roll
    except Exception as e:
        print(f"Error processing {midi_file}: {e}")
        return None


class MidiDatasetMulti(Dataset):
    def __init__(self, midi_dir, fs=100, fixed_length=1000, max_tracks=4):
        self.midi_files = find_all_midi_files(midi_dir)
        self.fs = fs
        self.fixed_length = fixed_length
        self.max_tracks = max_tracks
        self.data = []
        self._prepare_dataset()

    def _prepare_dataset(self):
        for midi_file in self.midi_files:
            multi_roll = midi_to_multi_piano_roll(midi_file, fs=self.fs,
                                                  max_tracks=self.max_tracks,
                                                  fixed_length=self.fixed_length)
            if multi_roll is not None:
                self.data.append(multi_roll)
        if len(self.data) > 0:
            self.data = np.array(self.data)
        else:
            print("❌ 没有有效 MIDI 数据")
        print(f"✅ 成功加载 {len(self.data)} 条 MIDI 数据")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)


# -----------------------------
# 主训练函数
# -----------------------------
def train_gan_music(midi_dir, epochs=50, batch_size=16, latent_dim=100, fs=100, fixed_length=1000, max_tracks=4):
    dataset = MidiDatasetMulti(midi_dir, fs=fs, fixed_length=fixed_length, max_tracks=max_tracks)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    if len(dataset) == 0:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample_shape = dataset.data[0].shape
    generator = Generator(latent_dim, sample_shape[0], sample_shape[1], sample_shape[2]).to(device)
    discriminator = Discriminator(sample_shape).to(device)

    # 使用 LSGAN 损失函数（MSE）
    criterion = nn.MSELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for i, real_data in enumerate(dataloader):
            real_data = real_data.to(device)

            # 使用 Label Smoothing：真实数据标签是 0.9 而不是 1.0
            valid = torch.ones(real_data.size(0), 1).to(device) * 0.9
            fake = torch.zeros(real_data.size(0), 1).to(device)

            # ---------------------
            #  训练 Discriminator
            # ---------------------
            if i % 2 == 0:
                optimizer_D.zero_grad()

                # real
                real_output = discriminator(real_data)
                real_loss = criterion(real_output, valid)

                # fake
                z = torch.randn(real_data.size(0), latent_dim).to(device)
                gen_data = generator(z)
                gen_data_noisy = gen_data + torch.randn_like(gen_data) * 0.01  # 加扰动
                fake_output = discriminator(gen_data_noisy.detach())
                fake_loss = criterion(fake_output, fake)

                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                optimizer_D.step()
            else:
                d_loss = torch.tensor(0.0)  # 记录方便打印

            # ---------------------
            #  训练 Generator（每个 batch 训练两次）
            # ---------------------
            # ✅ 使用 REINFORCE 风格 reward 引导 G 训练
            # 替换原来的 Generator loss 部分

            # ✅ 使用 REINFORCE 风格 reward 引导 G 训练
            # 替换原来的 Generator loss 部分

            for _ in range(2):  # G 每个 batch 训练两次
                # ✅ 使用 REINFORCE 风格 reward 引导 G 训练（加入 soft density reward + 激活 bias）
                # 替换原来的 Generator loss 部分

                optimizer_G.zero_grad()
                z = torch.randn(real_data.size(0), latent_dim).to(device)
                raw_output = generator(z)

                # 加 bias，提升激活概率，便于采样出非零 note
                probs = torch.sigmoid(raw_output + 0.2)
                m = torch.distributions.Bernoulli(probs)
                sampled = m.sample()

                total_g_loss = 0.0
                total_rhythm_reward = 0.0
                total_density_score = 0.0
                reward_scale = max(0.05 * (1.0 - epoch / epochs), 0.01)

                for b in range(sampled.size(0)):
                    # rhythm reward: 仍然基于 sample 的 numpy 判断
                    rhythm = reward_from_rhythm(sampled[b].detach().cpu().numpy())
                    total_rhythm_reward += rhythm

                    # density score: 使用概率的平均值（soft reward, 可微）
                    density_score = probs[b].mean()
                    total_density_score += density_score.item()

                    total_reward = 0.2 * rhythm + 1.0 * density_score  # rhythm 是 0~0.5，density 是 0~1
                    log_prob = m.log_prob(sampled[b]).mean()
                    g_loss = -log_prob * reward_scale * total_reward
                    g_loss.backward(retain_graph=True)
                    total_g_loss += g_loss.item()

                optimizer_G.step()

                if i % 10 == 0:
                    print(f"[Epoch {epoch + 1}/{epochs}] [Batch {i}/{len(dataloader)}] "
                          f"D_loss: {d_loss.item():.4f}  G_loss: {total_g_loss:.4f}  "
                          f"Reward(Rhythm/SoftDensity): {total_rhythm_reward:.2f} / {total_density_score:.2f}")

    # 保存模型
    models_dir = os.path.join(midi_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    torch.save(generator.state_dict(), os.path.join(models_dir, "generator_version4.pth"))
    torch.save(discriminator.state_dict(), os.path.join(models_dir, "discriminator_version4.pth"))
    print("✅ 模型已保存！")


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    midi_dir = os.path.join(base_dir, "fixed_midi")
    train_gan_music(midi_dir)
