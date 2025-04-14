# utils.py

import numpy as np

def detect_beat(piano_roll, fs=100, window=16):
    """
    简单节奏检测：计算每 window 个时间步内鼓点数量的周期性
    """
    drum_track = piano_roll[-1]  # 最后一轨为 drum
    activation = drum_track.sum(axis=0)  # shape: (time,)
    grouped = activation.reshape(-1, window).sum(axis=1)
    variance = np.var(grouped)  # 节奏波动越大越有节奏感
    return variance


def reward_from_rhythm(piano_roll):
    """
    给 Generator 的 reward，如果节奏性好则奖励
    """
    beat_strength = detect_beat(piano_roll)
    if beat_strength > 5.0:  # 阈值可调
        return 0.5
    return 0.0
