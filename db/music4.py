import torch
import pretty_midi
import numpy as np
import torch.nn as nn
import os


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


# 载入已经训练好的 Generator 模型
def load_generator(model_path, latent_dim=100, n_tracks=4, n_pitches=128, seq_len=256, device="cpu"):
    generator = Generator(latent_dim, n_tracks, n_pitches, seq_len).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()  # 设置为评估模式
    return generator


# 生成音乐
def generate_music(generator, latent_dim=100, device="cpu", beat_interval=8):
    z = torch.randn(1, latent_dim, device=device)  # 随机潜在向量
    with torch.no_grad():
        gen_sample = generator(z).squeeze(0).cpu().numpy()

    # 后处理：应用节奏mask + pitch限制
    processed_tracks = []
    for track_idx in range(gen_sample.shape[0]):
        processed = process_track(gen_sample[track_idx], track_idx, beat_interval=beat_interval,
                                  track_idx_config=track_idx)
        processed_tracks.append(processed)

    processed_tracks = np.stack(processed_tracks, axis=0)  # (n_tracks, 128, T)

    # 保存为 MIDI 文件
    save_path = "generated_music.mid"
    save_multitrack_pianoroll_as_midi(processed_tracks, save_path, fs=24)
    print(f"🎵 生成的音乐已保存到: {save_path}")


# 为每个轨道设置不同的乐器和节奏配置
def process_track(roll, track_idx, beat_interval=8, track_idx_config=None):
    roll = np.clip(roll, 0, 1)
    sparse = np.zeros_like(roll)
    T = roll.shape[1]

    # 节奏配置（根据 track_idx_config 设置不同的节奏）
    rhythm_configs = [
        {"interval": 48, "pattern": [1, 0, 0, 0]},  # 很慢
        {"interval": 64, "pattern": [0, 1, 0, 1]},  # 更慢
        {"interval": 96, "pattern": [1, 1, 0, 1]},  # 稀疏低音
        {"interval": 24, "pattern": [1, 1, 1, 1]}  # 极少的鼓点
    ]

    rhythm = rhythm_configs[track_idx_config]  # 选择当前轨道的节奏配置
    interval = rhythm["interval"]
    pattern = rhythm["pattern"]

    # 音符生成
    for t in range(0, T, interval):
        if pattern[t % len(pattern)] == 1:
            top_pitches = np.argsort(roll[:, t])[-1:]  # 选择概率最大的音符
            sparse[:, t][top_pitches] = 1

    return sparse


# 将多轨钢琴卷帘图保存为 MIDI 文件
def save_multitrack_pianoroll_as_midi(piano_roll, filename, fs=24):
    midi = pretty_midi.PrettyMIDI()

    # 为每个轨道分配不同的乐器配置
    instrument_configs = [
        {"program": 0, "name": "Acoustic Grand Piano", "is_drum": False},  # Melody
        {"program": 48, "name": "String Ensemble 1", "is_drum": False},  # Harmony
        {"program": 32, "name": "Electric Bass (finger)", "is_drum": False},  # Bass
        {"program": 118, "name": "Electronic Drum Kit", "is_drum": True},  # Drums
    ]

    for i, roll in enumerate(piano_roll):
        instrument_config = instrument_configs[i]
        instrument = pretty_midi.Instrument(program=instrument_config["program"], is_drum=instrument_config["is_drum"])
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
                        note = pretty_midi.Note(velocity=80, pitch=pitch, start=start / fs, end=end / fs)
                        instrument.notes.append(note)
                    active = False
            if active:
                note = pretty_midi.Note(velocity=80, pitch=pitch, start=start / fs, end=roll.shape[1] / fs)
                instrument.notes.append(note)
        midi.instruments.append(instrument)

    midi.write(filename)


# 主函数：生成音乐
if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "generated_fixed", "final_models", "generator_final.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = load_generator(model_path, device=device)
    generate_music(generator, device=device)
