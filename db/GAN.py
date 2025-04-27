# ================================
# 导入依赖
# ================================
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pretty_midi
import mido
from tqdm import tqdm
import matplotlib.pyplot as plt

##reward
def reward_from_rhythm(piano_roll, min_notes=1, max_notes=4):
    """
    奖励每个时间步活跃的音符数在合理范围内（有断有续）
    piano_roll: (n_tracks, 128, T)
    """
    score = 0.0
    T = piano_roll.shape[-1]
    for t in range(T):
        active = (piano_roll[..., t] > 0.5).sum()
        if active >= 1:
            score += 1.0
    return score / T

def reward_from_density(piano_roll, target_density=0.1, tolerance=0.05):
    """
    奖励音符密度在合理范围
    target_density: 理想的活跃率，比如10%
    """
    total_notes = np.prod(piano_roll.shape[:-1]) * piano_roll.shape[-1]
    active_notes = (piano_roll > 0.5).sum()
    actual_density = active_notes / total_notes
    # 奖励密度接近目标密度
    return max(0.0, 1.0 - abs(actual_density - target_density) / tolerance)

def reward_from_chords(piano_roll, fs=100):
    """
    Evaluate if notes form chords in each time frame.
    Reward frames where 3 or more notes are active simultaneously.
    """
    score = 0.0
    time_steps = piano_roll.shape[-1]
    for t in range(time_steps):
        active_notes = np.sum(piano_roll[:, t] > 0)
        if active_notes >= 3:
            score += 1.0
    return score / time_steps




# ================================
# MIDI工具函数
# ================================

def is_midi_valid(midi_file, max_tick_threshold=16000000, max_messages=500000):
    try:
        mid = mido.MidiFile(midi_file)
        total_messages = sum(len(track) for track in mid.tracks)
        if total_messages > max_messages:
            print(f"⚠️ 消息数超限 ({total_messages})，跳过 {midi_file}")
            return False
        for track in mid.tracks:
            for msg in track:
                if hasattr(msg, 'time') and msg.time > max_tick_threshold:
                    print(f"⚠️ 单条tick时间异常 ({msg.time})，跳过 {midi_file}")
                    return False
        return True
    except Exception as e:
        print(f"⚠️ mido解析失败，跳过: {midi_file}，错误信息: {e}")
        return False


def find_all_midi_files(root_dir):
    midi_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(('.mid', '.midi')):
                midi_files.append(os.path.join(dirpath, filename))
    return midi_files


def midi_to_multi_piano_roll(midi_file, fs=100, max_tracks=4, fixed_length=500):
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file)
        tracks = []
        for instrument in midi_data.instruments:
            roll = instrument.get_piano_roll(fs=fs) / 127.0
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


def reward_from_chords_multitrack(roll, min_tracks_with_notes=2, min_notes_per_frame=3):
    num_tracks, num_pitches, time_steps = roll.shape
    score = 0.0
    for t in range(time_steps):
        notes_per_track = [(roll[trk, :, t] > 0.5).sum() for trk in range(num_tracks)]
        if sum([n > 0 for n in notes_per_track]) >= min_tracks_with_notes and sum(
                notes_per_track) >= min_notes_per_frame:
            score += 1.0
    return score / time_steps


def save_pianoroll_as_midi(piano_roll, filename, fs=100):
    midi = pretty_midi.PrettyMIDI()
    for i, roll in enumerate(piano_roll):
        instrument = pretty_midi.Instrument(program=0)
        for pitch in range(128):
            active = False
            start = 0
            for t in range(roll.shape[1]):
                if roll[pitch, t] > 0 and not active:
                    active = True
                    start = t
                elif roll[pitch, t] == 0 and active:
                    end = t
                    note = pretty_midi.Note(velocity=100, pitch=pitch, start=start / fs, end=end / fs)
                    instrument.notes.append(note)
                    active = False
            if active:
                note = pretty_midi.Note(velocity=100, pitch=pitch, start=start / fs, end=roll.shape[1] / fs)
                instrument.notes.append(note)
        midi.instruments.append(instrument)
    midi.write(filename)


# ================================
# 数据集模块
# ================================

class MidiDatasetMulti(Dataset):
    def __init__(self, midi_dir, fs=100, fixed_length=500, max_tracks=4):
        self.midi_files = find_all_midi_files(midi_dir)
        self.fs = fs
        self.fixed_length = fixed_length
        self.max_tracks = max_tracks
        self.data = []
        self._prepare_dataset()

    def _prepare_dataset(self):
        for midi_file in self.midi_files:
            if not is_midi_valid(midi_file):
                continue
            multi_roll = midi_to_multi_piano_roll(midi_file, fs=self.fs, max_tracks=self.max_tracks,
                                                  fixed_length=self.fixed_length)
            if multi_roll is not None:
                self.data.append(multi_roll)
        if len(self.data) > 0:
            self.data = np.array(self.data)
        else:
            print("❗ 未找到任何有效的MIDI文件！")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return torch.tensor(sample, dtype=torch.float32)


# ================================
# 模型模块
# ================================

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, n_tracks=4, n_pitches=128, seq_len=256, n_layers=4, n_heads=8):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.n_tracks = n_tracks
        self.n_pitches = n_pitches
        self.seq_len = seq_len
        self.embed_dim = 512

        # 把latent z扩展成序列
        self.latent_to_seq = nn.Linear(latent_dim, self.embed_dim * seq_len)

        # 可学习的位置编码（位置感知）
        self.pos_embedding = nn.Parameter(torch.randn(seq_len, self.embed_dim))

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=n_heads,
            dim_feedforward=1024,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 输出映射到 piano roll
        self.output_layer = nn.Linear(self.embed_dim, n_tracks * n_pitches)

        # 初始化权重（更快收敛）
        self._initialize_weights()

    def forward(self, z):
        batch_size = z.size(0)
        x = self.latent_to_seq(z).view(batch_size, self.seq_len, self.embed_dim)
        x = x + self.pos_embedding.unsqueeze(0).to(x.device)
        x = self.transformer(x)
        x = self.output_layer(x)
        x = x.view(batch_size, self.seq_len, self.n_tracks, self.n_pitches)
        x = x.permute(0, 2, 3, 1)  # -> [B, n_tracks, n_pitches, T]
        return x

    def _initialize_weights(self):
        # 线性层用正态分布初始化，均值0，标准差0.02
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)



class Discriminator(nn.Module):
    def __init__(self, input_shape, d_model=512, nhead=8, num_layers=4):
        super(Discriminator, self).__init__()
        self.input_shape = input_shape
        self.c, self.h, self.w = input_shape
        self.seq_len = self.w
        self.feature_dim = self.c * self.h

        self.input_proj = nn.Linear(self.feature_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(self.seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=1024, dropout=0.1,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), self.c * self.h, self.w).transpose(1, 2)
        x = self.input_proj(x)
        x = x + self.pos_embedding.unsqueeze(0).to(x.device)
        x = self.transformer(x)
        x = x.mean(dim=1)
        validity = self.output_layer(x)
        return validity


# ================================
# 训练模块
# ================================

def train_musegan(midi_dir, epochs=100, batch_size=16, latent_dim=100, fs=100, fixed_length=256, max_tracks=4,
                  resume=False, generator_path="", discriminator_path="",
                  init_chord_weight=0.5, init_rhythm_weight=1.5, init_density_weight=1.5,
                  binarize_threshold=0.3, save_checkpoint_every=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用设备：{device}")

    dataset = MidiDatasetMulti(midi_dir, fs=fs, fixed_length=fixed_length, max_tracks=max_tracks)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if len(dataset) == 0:
        print("❗ 数据集为空")
        return

    sample_shape = dataset.data[0].shape
    generator = Generator(latent_dim=latent_dim, n_tracks=sample_shape[0], n_pitches=sample_shape[1],
                           seq_len=sample_shape[2]).to(device)
    discriminator = Discriminator(sample_shape).to(device)

    if resume:
        if os.path.exists(generator_path):
            generator.load_state_dict(torch.load(generator_path))
            print(f"✅ Loaded Generator from {generator_path}")
        if os.path.exists(discriminator_path):
            discriminator.load_state_dict(torch.load(discriminator_path))
            print(f"✅ Loaded Discriminator from {discriminator_path}")

    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0005, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0005, betas=(0.5, 0.999))

    all_d_losses = []
    all_g_losses = []
    all_rewards = []

    for epoch in range(epochs):
        # ====== 动态调整reward权重 ======
        if epoch < 10:
            chord_weight = 0.7
            rhythm_weight = 0.7
            density_weight = 0.6
        elif epoch < 20:
            chord_weight = 0.5
            rhythm_weight = 1.0
            density_weight = 1.0
        elif epoch < 30:
            chord_weight = 0.3
            rhythm_weight = 1.5
            density_weight = 1.5
        else:
            chord_weight = 0.2
            rhythm_weight = 2.0
            density_weight = 2.0
        # ================================

        progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]", ncols=140)

        for real_data in progress_bar:
            real_data = real_data.to(device)
            current_bs = real_data.size(0)

            valid = torch.ones(current_bs, 1, device=device) * 0.9
            fake = torch.zeros(current_bs, 1, device=device)

            # 训练 D
            optimizer_D.zero_grad()
            validity_real = discriminator(real_data)
            d_loss_real = adversarial_loss(validity_real, valid)

            z = torch.randn(current_bs, latent_dim, device=device)
            gen_data = generator(z).detach()
            gen_data = gen_data + 0.02 * torch.randn_like(gen_data)

            validity_fake = discriminator(gen_data)
            d_loss_fake = adversarial_loss(validity_fake, fake)

            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()

            # 训练 G
            for _ in range(2):
                optimizer_G.zero_grad()
                z = torch.randn(current_bs, latent_dim, device=device)
                gen_data = generator(z)
                validity_fake = discriminator(gen_data)

                chord_reward = 0.0
                rhythm_reward = 0.0
                density_reward = 0.0

                for b in range(gen_data.size(0)):
                    roll = gen_data[b].detach().cpu().numpy()
                    merged_roll = np.sum(roll, axis=0)
                    chord_reward += reward_from_chords(merged_roll)
                    rhythm_reward += reward_from_rhythm(merged_roll)
                    density_reward += reward_from_density(merged_roll)

                chord_reward /= gen_data.size(0)
                rhythm_reward /= gen_data.size(0)
                density_reward /= gen_data.size(0)

                g_loss = adversarial_loss(validity_fake, valid)
                g_loss -= chord_weight * chord_reward
                g_loss -= rhythm_weight * rhythm_reward
                g_loss -= density_weight * density_reward

                g_loss.backward()
                optimizer_G.step()

            all_d_losses.append(d_loss.item())
            all_g_losses.append(g_loss.item())
            all_rewards.append(chord_reward)

            progress_bar.set_postfix({
                'D_loss': f"{d_loss.item():.4f}",
                'G_loss': f"{g_loss.item():.4f}",
                'Chord_R': f"{chord_reward:.4f}",
                'Rhythm_R': f"{rhythm_reward:.4f}",
                'Density_R': f"{density_reward:.4f}"
            })

        # 每个epoch保存一首生成MIDI
        with torch.no_grad():
            z_sample = torch.randn(1, latent_dim, device=device)
            gen_sample = generator(z_sample).squeeze(0).cpu().numpy()
            binarized = (gen_sample > binarize_threshold).astype(np.uint8)
            output_dir = os.path.join(midi_dir, "generated_midis")
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"epoch{epoch+1:03d}.mid")
            save_pianoroll_as_midi(binarized, save_path)

        # 每5轮保存一次 checkpoint
        if (epoch + 1) % save_checkpoint_every == 0:
            models_dir = os.path.join(midi_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            torch.save(generator.state_dict(), os.path.join(models_dir, f"generator_epoch{epoch+1:03d}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(models_dir, f"discriminator_epoch{epoch+1:03d}.pth"))
            print(f"✅ Checkpoint saved at epoch {epoch+1}")

    # 最后绘制loss和reward曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(all_d_losses, label="D_loss")
    plt.title("Discriminator Loss")
    plt.xlabel("Batch Step")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(all_g_losses, label="G_loss")
    plt.title("Generator Loss")
    plt.xlabel("Batch Step")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(all_rewards, label="Chord Reward")
    plt.title("Chord Reward")
    plt.xlabel("Batch Step")
    plt.ylabel("Reward")
    plt.legend()

    plt.tight_layout()
    plt.show()



# ================================
# 主程序入口
# ================================

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    midi_dir = os.path.join(base_dir, "fixed_midi")
    train_musegan(
        midi_dir=midi_dir,
        epochs=50,
        batch_size=16,
        latent_dim=100,
        fs=100,
        fixed_length=256,
        max_tracks=4,
        init_chord_weight=0.5,
        init_rhythm_weight=1.5,
        init_density_weight=1.5
    )

