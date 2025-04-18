# train_gan_music.py

import os

import numpy as np
import pretty_midi
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model2 import Generator, Discriminator
from utils import reward_from_rhythm, reward_from_pitch_range, reward_from_silence, reward_from_density


def save_pianoroll_as_midi(piano_roll, filename, fs=100):
    """
    Convert a multi-track piano roll to a MIDI file and save it.

    Args:
        piano_roll (np.ndarray): Shape (num_tracks, 128, time_steps)
        filename (str): Output path
        fs (int): Frames per second (time resolution)
    """
    import pretty_midi
    midi = pretty_midi.PrettyMIDI()

    for i, roll in enumerate(piano_roll):
        instrument = pretty_midi.Instrument(program=0)
        velocity = 100  # constant velocity

        for pitch in range(128):
            active = False
            start = 0
            for t in range(roll.shape[1]):
                if roll[pitch, t] > 0 and not active:
                    active = True
                    start = t
                elif roll[pitch, t] == 0 and active:
                    end = t
                    start_time = start / fs
                    end_time = end / fs
                    note = pretty_midi.Note(velocity, pitch, start_time, end_time)
                    instrument.notes.append(note)
                    active = False
            if active:
                # Handle case where note continues till end
                end_time = roll.shape[1] / fs
                note = pretty_midi.Note(velocity, pitch, start / fs, end_time)
                instrument.notes.append(note)

        midi.instruments.append(instrument)

    midi.write(filename)

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

    criterion = nn.MSELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for i, real_data in enumerate(dataloader):
            real_data = real_data.to(device)

            valid = torch.ones(real_data.size(0), 1).to(device) * 0.9
            fake = torch.zeros(real_data.size(0), 1).to(device)

            # 训练 D
            if i % 2 == 0:
                optimizer_D.zero_grad()
                real_output = discriminator(real_data)
                real_loss = criterion(real_output, valid)

                z = torch.randn(real_data.size(0), latent_dim).to(device)
                gen_data, _ = generator(z)
                gen_data_noisy = gen_data + torch.randn_like(gen_data) * 0.01
                fake_output = discriminator(gen_data_noisy.detach())
                fake_loss = criterion(fake_output, fake)

                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                optimizer_D.step()
            else:
                d_loss = torch.tensor(0.0)

            # 训练 G
            for _ in range(2):  # G train twice
                optimizer_G.zero_grad()
                z = torch.randn(real_data.size(0), latent_dim).to(device)
                raw_output, frame_gate = generator(z)

                note_probs = torch.sigmoid(raw_output + 0.5)  # [B, 4, 128, T]
                noise_scale = max(0.3 * (1 - epoch / epochs), 0.05)
                exploration_noise = torch.randn_like(frame_gate) * noise_scale
                frame_gate = torch.clamp(frame_gate + exploration_noise, 0.0, 1.0)  # [B, 1, 1, T]

                # Apply silence gate
                probs = note_probs * frame_gate  # [B, 4, 128, T]

                # You can either use Bernoulli sampling or soft probs
                m = torch.distributions.Bernoulli(probs)
                sampled = m.sample()

                total_g_loss = 0.0
                total_rhythm_reward = 0.0
                total_density_score = 0.0
                total_pitch_range = 0.0
                total_silence = 0.0

                reward_scale = max(0.05 * (1.0 - epoch / epochs), 0.01)

                batch_g_loss = 0.0
                batch_loss_tensor = torch.tensor(0.0, device=device, requires_grad=True)

                for b in range(sampled.size(0)):
                    sample_np = sampled[b].detach().cpu().numpy()

                    rhythm = reward_from_rhythm(sample_np)
                    pitch_range = reward_from_pitch_range(sample_np)
                    silence = reward_from_silence(sample_np)
                    density_score = reward_from_density(sample_np, target_density=0.15)

                    total_rhythm_reward += rhythm
                    total_pitch_range += pitch_range
                    total_silence += silence
                    total_density_score += density_score

                    # if epoch < 5:
                    #     # Early phase: encourage rhythm & density
                    #     total_reward = (
                    #             0.5 * rhythm +
                    #             0.3 * pitch_range +
                    #             0.1 * silence +
                    #             0.6 * density_score
                    #     )
                    # else:
                    #     silence_weight = max(0.2 * (1 - epoch / epochs), 0.05)  # start at 0.2, decay to 0.05
                    #     # Balanced after warmup
                    #     total_reward = (
                    #             0.2 * rhythm +
                    #             0.4 * pitch_range +
                    #             silence_weight * silence +
                    #             0.4 * density_score
                    #     )

                    # Ideal targets
                    target_silence = 0.2
                    target_density = 0.15

                    # Penalty: distance from target
                    silence_penalty = abs(silence - target_silence)
                    density_penalty = abs(density_score - target_density)

                    # Reward: inverse of penalty
                    silence_bonus = max(0.0, 1.0 - silence_penalty * 5)
                    density_bonus = max(0.0, 1.0 - density_penalty * 5)

                    # Combine with base rewards
                    total_reward = (
                            0.2 * rhythm +
                            0.4 * pitch_range +
                            0.2 * silence_bonus +
                            0.2 * density_bonus
                    )

                    log_prob = m.log_prob(sampled[b]).mean()
                    loss = -log_prob * reward_scale * total_reward
                    batch_loss_tensor = batch_loss_tensor + loss

                    batch_g_loss += loss.item()

                # ONE backward call
                batch_loss_tensor.backward()
                total_g_loss = batch_g_loss

                optimizer_G.step()

                if i % 10 == 0:
                    actual_density = np.mean(sample_np > 0)
                    actual_silence_ratio = reward_from_silence(sample_np)
                    print(f"[Epoch {epoch + 1}/{epochs}] [Batch {i}/{len(dataloader)}] "
                          f"D_loss: {d_loss.item():.4f}  G_loss: {total_g_loss:.4f}  "
                          f"Reward(Rhythm/Pitch/Silence/Density): "
                          f"{total_rhythm_reward:.2f} / {total_pitch_range:.2f} / "
                          f"{actual_silence_ratio:.2f} / {actual_density:.2f}")

        # === Save generated MIDI ===
        with torch.no_grad():
            z_test = torch.randn(1, latent_dim).to(device)
            gen_out, gate = generator(z_test)
            note_probs = torch.sigmoid(gen_out + 0.5)
            gate = gate.clamp(0.0, 1.0)
            output = (note_probs * gate).squeeze(0).detach().cpu().numpy()
            sampled = (output > 0.3).astype(np.uint8)

            # Save to root-level "generated_midis" directory
            root_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(root_dir, "generated_midis")
            os.makedirs(output_dir, exist_ok=True)

            save_path = os.path.join(output_dir, f"epoch{epoch + 1}_batch{i}.mid")
            save_pianoroll_as_midi(sampled, save_path)
            print(f"🎵 Saved generated music to {save_path}")

    models_dir = os.path.join(midi_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    torch.save(generator.state_dict(), os.path.join(models_dir, "generator_version6.pth"))
    torch.save(discriminator.state_dict(), os.path.join(models_dir, "discriminator_version6.pth"))
    print("✅ 模型已保存！")


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    midi_dir = os.path.join(base_dir, "fixed_midi")
    train_gan_music(midi_dir)
