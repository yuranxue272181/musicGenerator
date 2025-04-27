import os
import torch
import numpy as np
import pretty_midi
from generate_music_version2_1 import Generator  # ✅ 用你的训练Generator

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
# 生成 latent 向量并生成 piano roll
# =========================
z = torch.randn(1, latent_dim, device=device)
with torch.no_grad():
    gen_sample = generator(z).squeeze(0).cpu().numpy()  # (n_tracks, 128, seq_len)

# =========================
# 定义节奏模板 + 四轨音色配置
# =========================

instrument_configs = [
    {"program": 80, "is_drum": False, "name": "Square Lead"},   # 方波旋律
    {"program": 81, "is_drum": False, "name": "Saw Lead"},      # 锯齿和声
    {"program": 38, "is_drum": False, "name": "Synth Bass"},    # 三角波低音
    {"program": 9, "is_drum": True, "name": "Noise Drums"}      # 鼓
]

rhythm_patterns = [
    list(range(0, seq_len, 16)),   # 第一轨每16步打一次
    list(range(4, seq_len, 24)),   # 第二轨4步偏移后每24步打一次
    list(range(0, seq_len, 32)),   # 第三轨每32步打一次
    [0, 8, 16, 24, 32, 40]         # 第四轨：特定鼓点
]

# =========================
# 稀疏化 piano roll（应用节奏）
# =========================

def sparsify_piano_roll(piano_roll, rhythm_patterns, max_notes_per_step=1):
    """
    按节奏稀疏 piano roll。
    """
    sparse_roll = np.zeros_like(piano_roll)
    for i, roll in enumerate(piano_roll):
        pattern = rhythm_patterns[i % len(rhythm_patterns)]
        roll = np.clip(roll, 0, 1)  # 避免异常值
        roll[95:, :] = 0  # 限制高频（避免刺耳）
        for t in pattern:
            if t >= roll.shape[1]:
                continue
            top_pitches = np.argsort(roll[:, t])[-max_notes_per_step:]
            sparse_roll[i, top_pitches, t] = 1
    return sparse_roll

sparse_roll = sparsify_piano_roll(gen_sample, rhythm_patterns, max_notes_per_step=1)

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

# 保存
output_dir = os.path.join(base_dir, "generated_final_midi")
os.makedirs(output_dir, exist_ok=True)

save_path = os.path.join(output_dir, "pixel_style_generated.mid")
save_multitrack_piano_roll_as_midi(sparse_roll, save_path, fs=100, configs=instrument_configs)

print(f"🎵 成功保存生成的像素风格MIDI！路径：{save_path}")
