import os
import torch
import numpy as np
import pretty_midi
from generate_music_version2_1 import Generator  # ✅ 根据你的文件名import

# =========================
# 参数设置
# =========================
latent_dim = 100
n_tracks = 4
n_pitches = 128
seq_len = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载训练好的 Generator
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "fixed_midi", "models", "generator_epoch050.pth")
generator = Generator(
    latent_dim=latent_dim,
    n_tracks=n_tracks,
    n_pitches=n_pitches,
    seq_len=seq_len
).to(device)

generator.load_state_dict(torch.load(model_path, map_location=device))
generator.eval()
print("✅ Generator 加载完成")

# =========================
# 定义节奏模板 + 四轨音色配置
# =========================

instrument_configs = [
    {"program": 80, "is_drum": False, "name": "Square Lead"},
    {"program": 81, "is_drum": False, "name": "Saw Lead"},
    {"program": 38, "is_drum": False, "name": "Synth Bass"},
    {"program": 9, "is_drum": True, "name": "Noise Drums"}
]

rhythm_patterns = [
    list(range(0, seq_len, 16)),   # 每16步主旋律
    list(range(4, seq_len, 24)),   # 每24步和声
    list(range(0, seq_len, 32)),   # 每32步低音
    [0, 8, 16, 24, 32, 40]         # 鼓点
]

# =========================
# 稀疏化 piano roll（应用节奏）
# =========================

def sparsify_piano_roll(piano_roll, rhythm_patterns, max_notes_per_step=1):
    sparse_roll = np.zeros_like(piano_roll)
    for i, roll in enumerate(piano_roll):
        pattern = rhythm_patterns[i % len(rhythm_patterns)]
        roll = np.clip(roll, 0, 1)
        roll[95:, :] = 0
        for t in pattern:
            if t >= roll.shape[1]:
                continue
            top_pitches = np.argsort(roll[:, t])[-max_notes_per_step:]
            sparse_roll[i, top_pitches, t] = 1
    return sparse_roll

# =========================
# 保存为 MIDI 文件
# =========================

def save_multitrack_piano_roll_as_midi(piano_roll, filename, fs=100, configs=None):
    midi = pretty_midi.PrettyMIDI()
    for i, roll in enumerate(piano_roll):
        cfg = configs[i % len(configs)] if configs else {"program": 0, "is_drum": False, "name": "Track"}

        instrument = pretty_midi.Instrument(program=cfg["program"], is_drum=cfg["is_drum"], name=cfg["name"])
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
                        instrument.notes.append(pretty_midi.Note(
                            velocity=80, pitch=pitch, start=start/fs, end=end/fs
                        ))
                    active = False
            if active:
                instrument.notes.append(pretty_midi.Note(
                    velocity=80, pitch=pitch, start=start/fs, end=roll.shape[1]/fs
                ))
        midi.instruments.append(instrument)

    midi.write(filename)

# =========================
# 批量生成10首，每首重复4遍保存
# =========================

output_dir = os.path.join(base_dir, "generated_final_midi_batch")
os.makedirs(output_dir, exist_ok=True)

num_songs = 10
repeat_times = 4

for idx in range(1, num_songs + 1):
    z = torch.randn(1, latent_dim, device=device)
    with torch.no_grad():
        gen_sample = generator(z).squeeze(0).cpu().numpy()

    sparse_roll = sparsify_piano_roll(gen_sample, rhythm_patterns, max_notes_per_step=1)

    # 重复4次
    repeated_roll = np.tile(sparse_roll, (1, 1, repeat_times))  # 时间维度重复

    save_path = os.path.join(output_dir, f"generated_song_{idx:02d}.mid")
    save_multitrack_piano_roll_as_midi(repeated_roll, save_path, fs=100, configs=instrument_configs)

    print(f"✅ 保存第 {idx} 首完成：{save_path}")

print(f"🎵 所有 {num_songs} 首音乐已生成，保存在：{output_dir}")
