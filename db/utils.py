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

def reward_from_density(piano_roll):
    active_notes = (piano_roll > 0.1).sum()
    total_notes = piano_roll.size
    density = active_notes / total_notes
    if 0.0005 < density < 0.2:  # 避免太稀或太密
        return 0.3
    return 0.0

def analyze_music(piano_roll):
    rhythm = reward_from_rhythm(piano_roll)
    density = reward_from_density(piano_roll)
    print(f"🎼 Rhythm Score: {rhythm:.2f} ({'✅' if rhythm > 0 else '⚠️'})",
          f"| Density Score: {density:.2f} ({'✅' if density > 0 else '⚠️'})")
