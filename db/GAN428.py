import os

import mido
import numpy as np
import pretty_midi
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import mido


# ===================
# 奖励函数
# ===================

class Generator(nn.Module):
    def __init__(self, latent_dim=100, n_tracks=4, n_pitches=128, seq_len=256):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, n_tracks * n_pitches * seq_len),
            nn.Sigmoid()  # 输出 piano roll
        )
        self.n_tracks = n_tracks
        self.n_pitches = n_pitches
        self.seq_len = seq_len

    def forward(self, z):
        out = self.fc(z)
        out = out.view(z.size(0), self.n_tracks, self.n_pitches, self.seq_len)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        c, h, w = input_shape
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * h * w, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def reward_from_beat_alignment(piano_roll, beat_interval=8):
    T = piano_roll.shape[-1]
    beats = np.arange(0, T, beat_interval)
    score = 0.0
    for t in beats:
        active = (piano_roll[..., t] > 0.5).sum()
        if active >= 1:
            score += 1.0
    return score / len(beats)

def reward_from_density(piano_roll, target_density=0.1, tolerance=0.05):
    total = np.prod(piano_roll.shape[:-1]) * piano_roll.shape[-1]
    active = (piano_roll > 0.5).sum()
    density = active / total
    return max(0.0, 1.0 - abs(density - target_density) / tolerance)

def reward_from_chords(piano_roll):
    T = piano_roll.shape[-1]
    score = 0.0
    for t in range(T):
        active_notes = (piano_roll[..., t] > 0.5).sum()
        if active_notes >= 3:
            score += 1.0
    return score / T

def train_musegan_fixed(
    dataset,
    generator,
    discriminator,
    epochs=50,
    batch_size=16,
    latent_dim=100,
    device="cpu",
    save_dir="generated_fixed",
    binarize_threshold=0.2,
    beat_interval=8
):
    os.makedirs(save_dir, exist_ok=True)

    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    all_d_losses = []
    all_g_losses = []

    for epoch in range(epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", ncols=120)

        for real_data in progress_bar:
            real_data = real_data.to(device)
            bs = real_data.size(0)

            valid = torch.ones(bs, 1, device=device) * 0.9
            fake = torch.zeros(bs, 1, device=device)

            # ====== Train Discriminator ======
            optimizer_D.zero_grad()
            real_pred = discriminator(real_data)
            d_real_loss = adversarial_loss(real_pred, valid)

            z = torch.randn(bs, latent_dim, device=device)
            fake_data = generator(z).detach()
            fake_data += 0.02 * torch.randn_like(fake_data)

            fake_pred = discriminator(fake_data)
            d_fake_loss = adversarial_loss(fake_pred, fake)

            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # ====== Train Generator ======
            for _ in range(2):
                optimizer_G.zero_grad()
                z = torch.randn(bs, latent_dim, device=device)
                gen_data = generator(z)

                validity = discriminator(gen_data)

                merged = torch.sum(gen_data, dim=1)
                merged = merged.detach().cpu().numpy()

                beat_reward = np.mean([reward_from_beat_alignment(m) for m in merged])
                density_reward = np.mean([reward_from_density(m) for m in merged])
                chord_reward = np.mean([reward_from_chords(m) for m in merged])

                g_loss = adversarial_loss(validity, valid)
                g_loss -= 0.5 * beat_reward
                g_loss -= 0.5 * density_reward
                g_loss -= 0.3 * chord_reward

                g_loss.backward()
                optimizer_G.step()

            progress_bar.set_postfix({
                "D_loss": f"{d_loss.item():.4f}",
                "G_loss": f"{g_loss.item():.4f}",
                "BeatR": f"{beat_reward:.2f}",
                "DensityR": f"{density_reward:.2f}",
                "ChordR": f"{chord_reward:.2f}"
            })

            all_d_losses.append(d_loss.item())
            all_g_losses.append(g_loss.item())

        # ===== 每个epoch保存生成MIDI ======
        save_sample_music(generator, latent_dim, epoch, save_dir, device, binarize_threshold, beat_interval)
        if (epoch + 1) % 5 == 0:
            models_dir = os.path.join(save_dir, "checkpoints")
            os.makedirs(models_dir, exist_ok=True)
            torch.save(generator.state_dict(), os.path.join(models_dir, f"generator_epoch{epoch + 1:03d}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(models_dir, f"discriminator_epoch{epoch + 1:03d}.pth"))
            print(f"✅ Saved checkpoint at epoch {epoch + 1}")

    final_models_dir = os.path.join(save_dir, "final_models")
    os.makedirs(final_models_dir, exist_ok=True)
    torch.save(generator.state_dict(), os.path.join(final_models_dir, "generator_final.pth"))
    torch.save(discriminator.state_dict(), os.path.join(final_models_dir, "discriminator_final.pth"))
    print("✅ Final models saved after training.")


def save_sample_music(generator, latent_dim, epoch, save_dir, device, binarize_threshold=0.2, beat_interval=8):
    z = torch.randn(1, latent_dim, device=device)
    with torch.no_grad():
        gen_sample = generator(z).squeeze(0).cpu().numpy()

    # ===================
    # 后处理：应用节奏mask + pitch限制
    # ===================
    processed_tracks = []
    for track_idx in range(gen_sample.shape[0]):
        processed = process_track(gen_sample[track_idx], track_idx, beat_interval=beat_interval)
        processed_tracks.append(processed)

    processed_tracks = np.stack(processed_tracks, axis=0)  # (n_tracks, 128, T)

    # ===================
    # 保存成MIDI
    # ===================
    save_path = os.path.join(save_dir, f"epoch{epoch+1:03d}.mid")
    save_multitrack_pianoroll_as_midi(processed_tracks, save_path, fs=24)

    print(f"🎵 Saved MIDI at: {save_path}")

def process_track(roll, track_idx, beat_interval=8):
    """
    不同轨道应用不同pitch范围 + 节奏稀疏化
    """
    roll = np.clip(roll, 0, 1)
    sparse = np.zeros_like(roll)
    T = roll.shape[1]

    if track_idx == 0:
        pitch_range = slice(60, 96)  # Melody: 中高音
    elif track_idx == 1:
        pitch_range = slice(48, 72)  # Harmony: 中音
    elif track_idx == 2:
        pitch_range = slice(30, 50)  # Bass: 低音
    else:
        pitch_range = slice(35, 45)  # Drum (kick/snare)

    for t in range(0, T, beat_interval):
        top_pitches = np.argsort(roll[pitch_range, t])[-1:]
        sparse[pitch_range, t][top_pitches] = 1

    return sparse

def save_multitrack_pianoroll_as_midi(piano_roll, filename, fs=24):
    midi = pretty_midi.PrettyMIDI()
    programs = [0, 48, 32, 118]  # Piano, Strings, Bass, Electronic Drums
    is_drums = [False, False, False, True]

    for i, roll in enumerate(piano_roll):
        instrument = pretty_midi.Instrument(program=programs[i], is_drum=is_drums[i])
        for pitch in range(128):
            active = False
            start = 0
            for t in range(roll.shape[1]):
                if roll[pitch, t] > 0 and not active:
                    active = True
                    start = t
                elif roll[pitch, t] == 0 and active:
                    end = t
                    if start < end:
                        note = pretty_midi.Note(velocity=80, pitch=pitch, start=start/fs, end=end/fs)
                        instrument.notes.append(note)
                    active = False
            if active:
                note = pretty_midi.Note(velocity=80, pitch=pitch, start=start/fs, end=roll.shape[1]/fs)
                instrument.notes.append(note)
        midi.instruments.append(instrument)

    midi.write(filename)


def find_all_midi_files(root_dir):
    midi_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(('.mid', '.midi')):
                midi_files.append(os.path.join(dirpath, filename))
    return midi_files


from torch.utils.data import Dataset
class MidiDatasetMulti(Dataset):
    def __init__(self, midi_dir, fs=24, fixed_length=256, max_tracks=4):
        self.midi_files = find_all_midi_files(midi_dir)  # 🔥 直接用外面的
        self.fs = fs
        self.fixed_length = fixed_length
        self.max_tracks = max_tracks
        self.data = []
        self._prepare_dataset()

    def is_midi_valid(self, midi_file, max_tick_threshold=16000000, max_messages=500000):
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

    def midi_to_multi_piano_roll(self, midi_file):
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_file)
            tracks = []
            for instrument in midi_data.instruments:
                roll = instrument.get_piano_roll(fs=self.fs) / 127.0
                tracks.append(roll)
                if len(tracks) >= self.max_tracks:
                    break
            if len(tracks) < self.max_tracks:
                max_time = max((r.shape[1] for r in tracks), default=0)
                for _ in range(self.max_tracks - len(tracks)):
                    tracks.append(np.zeros((128, max_time)))
            processed = []
            for roll in tracks:
                if roll.shape[1] < self.fixed_length:
                    pad_width = self.fixed_length - roll.shape[1]
                    roll = np.pad(roll, ((0, 0), (0, pad_width)), mode='constant')
                else:
                    roll = roll[:, :self.fixed_length]
                processed.append(roll)
            multi_roll = np.stack(processed, axis=0)
            return multi_roll
        except Exception as e:
            print(f"Error processing {midi_file}: {e}")
            return None

    def _prepare_dataset(self):
        print(f"📦 正在加载 {len(self.midi_files)} 个MIDI文件...")
        for midi_file in tqdm(self.midi_files, desc="加载MIDI数据", ncols=100):
            if not self.is_midi_valid(midi_file):
                continue
            multi_roll = self.midi_to_multi_piano_roll(midi_file)
            if multi_roll is not None:
                self.data.append(multi_roll)
        if len(self.data) > 0:
            self.data = np.array(self.data)
            print(f"✅ 成功加载了 {len(self.data)} 个有效的MIDI样本。")
        else:
            print("❗ 未找到任何有效的MIDI文件！")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return torch.tensor(sample, dtype=torch.float32)

if __name__ == "__main__":

    # ===================
    # 开始主流程
    # ===================
    base_dir = os.path.dirname(__file__)
    midi_dir = os.path.join(base_dir, "POP909")  # 🔵 确保这里是你的正确MIDI文件夹
    dataset = MidiDatasetMulti(midi_dir)

    # ✅ 训练过程中生成的音乐保存到这里
    save_dir = os.path.join(base_dir, "generated_from_pop909")
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    latent_dim = 100
    n_tracks = 4
    n_pitches = 128
    seq_len = 256

    generator = Generator(latent_dim, n_tracks, n_pitches, seq_len).to(device)
    discriminator = Discriminator((n_tracks, n_pitches, seq_len)).to(device)

    train_musegan_fixed(
        dataset=dataset,
        generator=generator,
        discriminator=discriminator,
        epochs=50,
        batch_size=16,
        latent_dim=latent_dim,
        device=device,
        save_dir=os.path.join(base_dir, "generated_fixed"),
        binarize_threshold=0.2,
        beat_interval=8
    )
