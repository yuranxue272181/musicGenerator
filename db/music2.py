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
# 音色配置
# =========================

#1
# instrument_configs = [
#     {"program": 80, "is_drum": False, "name": "Square Lead"},  # 主旋律
#     {"program": 81, "is_drum": False, "name": "Saw Lead"},     # 和声
#     {"program": 38, "is_drum": False, "name": "Synth Bass"},   # 低音
#     {"program": 9, "is_drum": True, "name": "Noise Drums"}     # 鼓
# ]

#2
# instrument_configs = [
#     {"program": 0, "is_drum": False, "name": "Acoustic Grand Piano"},  # 主旋律：钢琴
#     {"program": 48, "is_drum": False, "name": "String Ensemble 1"},     # 和声：弦乐
#     {"program": 32, "is_drum": False, "name": "Acoustic Bass"},         # 低音：原声低音
#     {"program": 0, "is_drum": True, "name": "Drum Kit"}                 # 鼓：标准鼓组
# ]

#3
# instrument_configs = [
#     {"program": 80, "is_drum": False, "name": "Synth Lead 1"},
#     {"program": 88, "is_drum": False, "name": "Pad 1 (New Age)"},
#     {"program": 38, "is_drum": False, "name": "Synth Bass 1"},
#     {"program": 0, "is_drum": True, "name": "Standard Drum Kit"}
# ]

#4
# instrument_configs = [
#     {"program": 40, "is_drum": False, "name": "Violin"},       # 主旋律：小提琴
#     {"program": 42, "is_drum": False, "name": "Cello"},        # 和声：大提琴
#     {"program": 43, "is_drum": False, "name": "Contrabass"},   # Bass：低音提琴
#     {"program": 48, "is_drum": True, "name": "Orchestral Percussion"}  # Drum：交响打击乐
# ]

#5
instrument_configs = [
    {"program": 8, "is_drum": False, "name": "Celesta"},            # 主旋律：钟琴（闪光）
    {"program": 52, "is_drum": False, "name": "Choir Aahs"},         # 和声：人声合唱
    {"program": 39, "is_drum": False, "name": "Synth Bass 2"},       # 低音：电子低音
    {"program": 118, "is_drum": True, "name": "Synth Drum"}          # 打击乐：电子打击
]
rhythm_settings = {
    0: {"interval": 8, "pitch_range": (72, 96)},   # Celesta高音区
    1: {"interval": 16, "pitch_range": (60, 72)},  # Choir中高音区
    2: {"interval": 32, "pitch_range": (36, 48)},  # Synth Bass低频
    3: {"pattern": [0, 48, 96, 144, 192, 240]}      # 打击乐稀疏点缀
}


def process_track(roll, track_idx):
    roll = np.clip(roll, 0, 1)
    sparse = np.zeros_like(roll)
    T = roll.shape[1]

    settings = rhythm_settings[track_idx]

    if track_idx in [0, 1, 2]:  # 主旋律、和声、Bass
        interval = settings["interval"]
        pitch_start, pitch_end = settings["pitch_range"]
        for t in range(0, T, interval):
            pitch_range = slice(pitch_start, pitch_end)
            top = np.argsort(roll[pitch_range, t])[-1:]
            sparse[pitch_range, t][top] = 1

    elif track_idx == 3:  # 鼓
        pattern = settings["pattern"]
        for t in pattern:
            if t < T:
                sparse[36, t] = 1  # GM标准Kick Drum (36)

    return sparse



#4
# def process_track(roll, track_idx):
#     roll = np.clip(roll, 0, 1)
#     sparse = np.zeros_like(roll)
#     T = roll.shape[1]
#
#     if track_idx == 0:
#         # 小提琴主旋律：每4-8步随机激活一次，pitch范围自由一些
#         t = 0
#         while t < T:
#             step = np.random.choice([4, 6, 8])
#             if t < T:
#                 top = np.argsort(roll[:, t])[-1:]
#                 sparse[top, t] = 1
#             t += step
#
#     elif track_idx == 1:
#         # 大提琴和声：每16步选2个音，中音区
#         mid_range = slice(48, 72)
#         for t in range(0, T, 16):
#             top = np.argsort(roll[mid_range, t])[-2:]
#             sparse[mid_range, t][top] = 1
#
#     elif track_idx == 2:
#         # 低音提琴Bass：每32步选1个低频音
#         low_range = slice(30, 50)
#         for t in range(0, T, 32):
#             top = np.argsort(roll[low_range, t])[-1:]
#             sparse[low_range, t][top] = 1
#
#     elif track_idx == 3:
#         # 打击乐：鼓点更稀疏，比如每32到48步敲一次
#         drum_times = list(range(0, T, np.random.choice([32, 40, 48])))
#         for t in drum_times:
#             if t < T:
#                 sparse[36, t] = 1  # 36号是标准Kick
#
#     return sparse
#
#


# =========================
# 不同轨道应用不同旋律规则
# =========================

# def process_track(roll, track_idx):
#     roll = np.clip(roll, 0, 1)
#     sparse = np.zeros_like(roll)
#     T = roll.shape[1]
#
#     if track_idx == 0:
#         # 主旋律：每8步挑最强1个
#         for t in range(0, T, 8):
#             top = np.argsort(roll[:, t])[-1:]
#             sparse[top, t] = 1
#
#     elif track_idx == 1:
#         # 和声：每16步挑2个，pitch集中在主旋律中音区
#         for t in range(0, T, 16):
#             mid_pitch_range = slice(48, 72)
#             top = np.argsort(roll[mid_pitch_range, t])[-2:]
#             sparse[mid_pitch_range, t][top] = 1
#
#     elif track_idx == 2:
#         # Bass：低音区，每32步一个根音
#         for t in range(0, T, 32):
#             low_pitch_range = slice(30, 50)
#             top = np.argsort(roll[low_pitch_range, t])[-1:]
#             sparse[low_pitch_range, t][top] = 1
#
#     elif track_idx == 3:
#         # Drum：固定鼓点
#         drum_pattern = [0, 8, 16, 24, 32, 40, 48, 56]
#         for t in drum_pattern:
#             if t < T:
#                 sparse[36, t] = 1  # Kick drum (标准GM 36)
#
#     return sparse

# =========================
# 保存为MIDI
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

output_dir = os.path.join(base_dir, "generated_final_midi_batch_melody")
os.makedirs(output_dir, exist_ok=True)

num_songs = 10
repeat_times = 4

for idx in range(1, num_songs + 1):
    z = torch.randn(1, latent_dim, device=device)
    with torch.no_grad():
        gen_sample = generator(z).squeeze(0).cpu().numpy()

    processed_tracks = []
    for i in range(gen_sample.shape[0]):
        processed = process_track(gen_sample[i], i)
        processed_tracks.append(processed)

    processed_tracks = np.stack(processed_tracks, axis=0)  # (4, 128, T)

    # 重复
    repeated_roll = np.tile(processed_tracks, (1, 1, repeat_times))

    save_path = os.path.join(output_dir, f"melody_song_{idx:02d}.mid")
    save_multitrack_piano_roll_as_midi(repeated_roll, save_path, fs=100, configs=instrument_configs)

    print(f"✅ 保存第 {idx} 首完成：{save_path}")

print(f"🎵 所有 {num_songs} 首主旋律版音乐已生成，保存在：{output_dir}")
