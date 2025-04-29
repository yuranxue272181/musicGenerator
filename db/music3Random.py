import os
import torch
import numpy as np
import pretty_midi
from generate_music_version2_1 import Generator  # 你的 Generator 模型

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
model_path = os.path.join(base_dir,"generated_from_pop909", "generator_epoch050.pth")
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
# 音色配置（梦幻氛围）
# =========================

#1 梦幻
# instrument_configs = [
#     {"program": 8, "is_drum": False, "name": "Celesta"},
#     {"program": 52, "is_drum": False, "name": "Choir Aahs"},
#     {"program": 39, "is_drum": False, "name": "Synth Bass 2"},
#     {"program": 118, "is_drum": True, "name": "Synth Drum"}
# ]

# instrument_configs = [
#     {"program": 5, "is_drum": False, "name": "Electric Piano 1"},  # 主旋律: Electric Piano
#     {"program": 48, "is_drum": False, "name": "Strings Ensemble"},  # 和声: Strings
#     {"program": 33, "is_drum": False, "name": "Fingered Bass"},     # Bass: Fingered Bass
#     {"program": 117, "is_drum": True, "name": "Percussion Kit"}     # Drums: Percussion instead of Synth Drum
# ]

# 节奏设置
# rhythm_settings = {
#     0: {"interval": 6, "pitch_range": (72, 96)},    # 主旋律：每6步1次（大约每0.06秒一次）
#     1: {"interval": 12, "pitch_range": (60, 72)},   # 和声：每12步1次
#     2: {"interval": 16, "pitch_range": (36, 52)},   # Bass：每16步一个低音根音
#     3: {"pattern": [0, 32, 64, 96, 128, 160, 192, 224]}  # 鼓点稀疏，4拍一次
# }

# rhythm_settings = {
#     0: {"interval": 12, "pitch_range": (72, 96)},
#     1: {"interval": 24, "pitch_range": (60, 72)},
#     2: {"interval": 32, "pitch_range": (36, 52)},
#     3: {"pattern": [0, 64, 128, 192]}

#
# # 节奏设置
# rhythm_settings = {
#     0: {"interval": 16, "pitch_range": (72, 96)},   # 主旋律：每16步一次（更悠闲）
#     1: {"interval": 32, "pitch_range": (60, 72)},   # 和声：每32步一次
#     2: {"interval": 48, "pitch_range": (36, 52)},   # Bass：每48步一个低音
#     3: {"pattern": [0, 64, 128, 192]}               # 鼓：每64步打一下，等于两小节一次
# }

#2 电子
# instrument_configs = [
#     {"program": 81, "is_drum": False, "name": "Lead 2 (Sawtooth)"},    # 锯齿Lead，非常典型的电子旋律音色
#     {"program": 86, "is_drum": False, "name": "Pad 2 (Warm)"},         # 暖音Pad，背景铺垫
#     {"program": 38, "is_drum": False, "name": "Synth Bass 1"},         # 合成器Bass，动感
#     {"program": 118, "is_drum": True, "name": "Synth Drum"}            # 合成器鼓组
# ]
#
# rhythm_settings = {
#     0: {"interval": 12, "pitch_range": (72, 96)},    # 主旋律，轻快但不太密
#     1: {"interval": 24, "pitch_range": (48, 72)},    # 背景Pad，慢速铺垫
#     2: {"interval": 16, "pitch_range": (36, 52)},    # Bass，每小节两次
#     3: {"pattern": [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240]}  # 鼓密集打点
# }

#3 rock
# instrument_configs = [
#     {"program": 29, "is_drum": False, "name": "Overdriven Guitar"},  # 过载吉他
#     {"program": 30, "is_drum": False, "name": "Distortion Guitar"}, # 失真吉他
#     {"program": 33, "is_drum": False, "name": "Fingered Bass"},      # 电贝斯
#     {"program": 118, "is_drum": True, "name": "Synth Drum"}          # 合成鼓，模拟摇滚鼓
# ]
#
# rhythm_settings = {
#     0: {"interval": 8, "pitch_range": (64, 80)},    # 主旋律吉他，比较密集
#     1: {"interval": 16, "pitch_range": (60, 76)},   # 副吉他
#     2: {"interval": 24, "pitch_range": (40, 52)},   # Bass慢一点
#     3: {"pattern": [0, 32, 64, 96, 128, 160, 192, 224]}  # 鼓点稳定，摇滚感
# }

#4 ambient
instrument_configs = [
    {"program": 89, "is_drum": False, "name": "Pad 0 (New Age)"},    # 新世纪Pad
    {"program": 91, "is_drum": False, "name": "Pad 2 (Warm)"},       # 暖Pad
    {"program": 96, "is_drum": False, "name": "FX 1 (Rain)"},        # 特效Rain
    {"program": 122, "is_drum": True, "name": "Percussive Organ"}    # 柔和打击
]

rhythm_settings = {
    0: {"interval": 48, "pitch_range": (65, 85)},    # 很慢
    1: {"interval": 64, "pitch_range": (50, 70)},    # 更慢
    2: {"interval": 96, "pitch_range": (40, 60)},    # 稀疏低音
    3: {"pattern": [0, 128, 256]}                    # 极少的鼓点
}



# =========================
# 处理每轨稀疏旋律
# =========================
# def process_track(roll, track_idx):
#     roll = np.clip(roll, 0, 1)
#     sparse = np.zeros_like(roll)
#     T = roll.shape[1]
#     settings = rhythm_settings[track_idx]
#
#     if track_idx in [0, 1, 2]:
#         interval = settings["interval"]
#         pitch_start, pitch_end = settings["pitch_range"]
#         for t in range(0, T, interval):
#             pitch_range = slice(pitch_start, pitch_end)
#             if track_idx == 1:
#                 top = np.argsort(roll[pitch_range, t])[-3:]
#                 sparse[pitch_range, t][top] = 1
#             else:
#                 top = np.argsort(roll[pitch_range, t])[-1:]
#                 sparse[pitch_range, t][top] = 1
#
#     elif track_idx == 3:
#         pattern = settings["pattern"]
#         for t in pattern:
#             if t < T:
#                 sparse[36, t] = 1
#
#     return sparse

def process_track(roll, track_idx):
    roll = np.clip(roll, 0, 1)
    sparse = np.zeros_like(roll)
    T = roll.shape[1]
    settings = rhythm_settings[track_idx]
    rng = np.random.default_rng()

    if track_idx in [0, 1, 2]:
        interval = settings["interval"]
        pitch_start, pitch_end = settings["pitch_range"]
        for t in range(0, T, interval):
            # Add slight timing jitter
            jitter = rng.integers(-2, 3)  # jitter by ±2 steps max
            t_jittered = min(max(t + jitter, 0), T - 1)
            pitch_range = slice(pitch_start, pitch_end)
            roll_slice = roll[pitch_range, t_jittered]

            if np.sum(roll_slice) > 0:
                top_indices = np.argsort(roll_slice)[-3:]  # top 3 candidates
                chosen_idx = rng.choice(top_indices)
                sparse[pitch_range, t_jittered][chosen_idx] = 1

    elif track_idx == 3:
        pattern = settings["pattern"]
        for t in pattern:
            # Add swing to drums (later beats slightly delayed)
            swing = rng.integers(0, 5) if t >= 64 else 0
            t_swinged = min(t + swing, T - 1)
            sparse[36, t_swinged] = 1  # simple kick drum at C2 (MIDI 36)

    return sparse

# =========================
# 保存为MIDI
# =========================
# def save_multitrack_piano_roll_as_midi(piano_roll, filename, fs=100, configs=None):
#     midi = pretty_midi.PrettyMIDI()
#     for i, roll in enumerate(piano_roll):
#         cfg = configs[i % len(configs)] if configs else {"program": 0, "is_drum": False, "name": "Track"}
#         instrument = pretty_midi.Instrument(program=cfg["program"], is_drum=cfg["is_drum"], name=cfg["name"])
#         for pitch in range(128):
#             active = False
#             start = 0
#             for t in range(roll.shape[1]):
#                 if roll[pitch, t] > 0 and not active:
#                     active = True
#                     start = t
#                 elif roll[pitch, t] == 0 and active:
#                     end = t
#                     if start < end:
#                         instrument.notes.append(pretty_midi.Note(
#                             velocity=np.random.randint(60, 90),
#                             pitch=pitch,
#                             start=start/fs,
#                             end=end/fs
#                         ))
#                     active = False
#             if active:
#                 instrument.notes.append(pretty_midi.Note(
#                     velocity=np.random.randint(60, 90),
#                     pitch=pitch,
#                     start=start/fs,
#                     end=roll.shape[1]/fs
#                 ))
#         midi.instruments.append(instrument)
#
#     midi.write(filename)

def save_multitrack_piano_roll_as_midi(piano_roll, filename, fs=100, configs=None):
    midi = pretty_midi.PrettyMIDI()
    rng = np.random.default_rng()

    for i, roll in enumerate(piano_roll):
        cfg = configs[i % len(configs)] if configs else {"program": 0, "is_drum": False, "name": "Track"}
        instrument = pretty_midi.Instrument(program=cfg["program"], is_drum=cfg["is_drum"], name=cfg["name"])

        # Set base velocity range depending on track type
        if i == 0:  # Melody
            velocity_range = (70, 100)
        elif i == 1:  # Harmony
            velocity_range = (50, 80)
        elif i == 2:  # Bass
            velocity_range = (60, 85)
        elif i == 3:  # Drums
            velocity_range = (65, 95)
        else:
            velocity_range = (60, 90)

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
                        velocity = rng.integers(*velocity_range)
                        instrument.notes.append(pretty_midi.Note(
                            velocity=velocity,
                            pitch=pitch,
                            start=start / fs,
                            end=end / fs
                        ))
                    active = False
            if active:
                velocity = rng.integers(*velocity_range)
                instrument.notes.append(pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch,
                    start=start / fs,
                    end=roll.shape[1] / fs
                ))
        midi.instruments.append(instrument)

    midi.write(filename)

# =========================
# 只生成一首音乐，拼接4遍
# =========================
output_dir = os.path.join(base_dir, "generated_final_midi_batch_single_repeat4")
os.makedirs(output_dir, exist_ok=True)

# 只生成一次
z = torch.randn(1, latent_dim, device=device)
with torch.no_grad():
    gen_sample = generator(z).squeeze(0).cpu().numpy()

processed_tracks = []
for i in range(gen_sample.shape[0]):
    processed = process_track(gen_sample[i], i)
    processed_tracks.append(processed)

processed_tracks = np.stack(processed_tracks, axis=0)  # (4, 128, T)

# 🔥 重复4次拼接
repeated_tracks = np.tile(processed_tracks, (1, 1, 4))  # (4, 128, T*4)

save_path = os.path.join(output_dir, "dreamy_song_repeat4.mid")
save_multitrack_piano_roll_as_midi(repeated_tracks, save_path, fs=100, configs=instrument_configs)

print(f"🎵 已成功生成并保存梦幻拼接版音乐：{save_path}")
