import numpy as np

def detect_beat(piano_roll, fs=100, window=16):
    """
    简单节奏检测：计算每 window 个时间步内鼓点数量的周期性
    """
    try:
        drum_track = piano_roll[-1]  # 最后一轨为 drum
        activation = drum_track.sum(axis=0)  # shape: (time,)
        time_len = len(activation)

        # 自动补齐为 window 的倍数
        remainder = time_len % window
        if remainder != 0:
            pad_len = window - remainder
            activation = np.pad(activation, (0, pad_len), mode='constant')

        grouped = activation.reshape(-1, window).sum(axis=1)
        variance = np.var(grouped)  # 节奏波动越大越有节奏感
        return variance
    except Exception as e:
        print(f"❌ detect_beat 错误: {e}")
        return 0.0


def reward_from_rhythm(piano_roll):
    """
    给 Generator 的 reward，如果节奏性好则奖励
    """
    beat_strength = detect_beat(piano_roll)
    if beat_strength > 5.0:  # 阈值可调
        return 0.5
    return 0.0
