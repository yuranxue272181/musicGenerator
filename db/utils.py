import numpy as np

def detect_beat(piano_roll, fs=100, window=16):
    try:
        drum_track = piano_roll[-1]  # 最后一轨是 drum
        activation = drum_track.sum(axis=0)
        time_len = len(activation)

        # 自动补齐为 window 的倍数
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
    """
    更严格的节奏性检测：variance > 20 才给奖励
    奖励值不再是固定的，而是根据节奏强度线性增长（最多 0.5）
    """
    beat_strength = detect_beat(piano_roll)
    if beat_strength > 50.0:
        # 奖励为 [0.0, 0.5] 之间，随节奏强度线性增长
        reward = min(0.5, (beat_strength - 50.0) / 50.0)
        return reward
    return 0.0
