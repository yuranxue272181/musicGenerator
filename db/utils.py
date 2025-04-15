# ✅ utils.py 中加入新的奖励机制
import numpy as np

def detect_beat(piano_roll, fs=100, window=16):
    try:
        drum_track = piano_roll[-1]  # 最后一轨默认是 Drum
        activation = drum_track.sum(axis=0)
        time_len = len(activation)
        remainder = time_len % window
        if remainder != 0:
            pad_len = window - remainder
            activation = np.pad(activation, (0, pad_len), mode='constant')
        grouped = activation.reshape(-1, window).sum(axis=1)
        variance = np.var(grouped)
        return variance
    except Exception as e:
        print(f"❌ detect_beat 错误: {e}")
        return 0.0

def reward_from_rhythm(piano_roll):
    beat_strength = detect_beat(piano_roll)
    if beat_strength < 80:
        return 0.0
    elif beat_strength >= 150:
        return 0.5
    else:
        return (beat_strength - 80.0) / 70.0 * 0.5

def reward_from_density(piano_roll, target_density=0.15):
    # piano_roll: (tracks, 128, T)
    actual_density = np.mean(piano_roll > 0)
    return 1.0 - abs(actual_density - target_density)



def analyze_music(piano_roll):
    instrument_names = ["Flute", "Violin", "Bass", "Drums"]
    threshold = 0.1  # 跟 midi 生成中的阈值一致

    for idx, channel in enumerate(piano_roll):
        active_notes = (channel > threshold).sum()
        status = "✅ Has notes" if active_notes > 0 else "❌ Silent"
        print(f"Channel {idx} ({instrument_names[idx]}): {status} (Active notes: {active_notes})")

# utils.py

def reward_from_pitch_range(piano_roll):
    # piano_roll: (tracks, 128, T)
    pitch_range = 0
    for track in piano_roll:
        pitches = np.where(track > 0)
        if pitches[0].size > 0:
            pitch_range += pitches[0].max() - pitches[0].min()
    return pitch_range / (len(piano_roll) * 128)  # normalize to 0~1


def reward_from_silence(piano_roll):
    # 统计时间上全为0的比例
    silence_frames = 0
    total_frames = piano_roll.shape[2]
    for t in range(total_frames):
        if np.sum(piano_roll[:, :, t]) == 0:
            silence_frames += 1
    return silence_frames / total_frames  # 比例越大越好



