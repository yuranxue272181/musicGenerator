import os
import torch
import numpy as np
import pretty_midi
from torch import nn

# ==== Generator 定义（保持不变）====
class Generator(nn.Module):
    def __init__(self, latent_dim, output_shape):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, np.prod(output_shape)),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(z.size(0), *self.output_shape)
        return out


def sparsify_roll(roll, max_notes_per_frame=2, beat_step=4):
    """
    roll: shape (128, time)
    限制每帧最多激活 max_notes_per_frame 个 note，
    只保留 beat_step 间隔的时间帧（模拟节拍）
    """
    time_steps = roll.shape[1]
    sparse = np.zeros_like(roll)

    for t in range(0, time_steps):
        if t % beat_step != 0:
            continue  # 只保留节奏点

        top_pitches = np.argsort(roll[:, t])[-max_notes_per_frame:]
        sparse[top_pitches, t] = 1

    return sparse.astype(np.uint8)

# ==== Piano roll 转 MIDI ====
def piano_roll_to_midi(piano_roll, fs=100):
    """
    piano_roll: shape (tracks, 128, time)
    返回节奏稀疏优化的 pretty_midi 对象
    """
    midi = pretty_midi.PrettyMIDI()
    time_steps = piano_roll.shape[2]

    # 乐器配置（更真实）
    instrument_configs = [
        {"program": 0,   "is_drum": False, "name": "Piano",  "beat": 8,  "notes": 3, "shift": 0},  # 每拍 1 和 3 撞击，模拟和弦
        {"program": 24,  "is_drum": False, "name": "Guitar", "beat": 12, "notes": 2, "shift": 2},  # 模拟吉他强拍落点
        {"program": 33,  "is_drum": False, "name": "Bass",   "beat": 16, "notes": 1, "shift": 4},  # 每小节根音节奏点（beat 1, 3）
        {"program": 0,   "is_drum": True,  "name": "Drums",  "beat": 4,  "notes": 3, "shift": 1},  # 模拟 kick/snare 在节拍上
    ]

    for i, roll in enumerate(piano_roll):
        cfg = instrument_configs[i % len(instrument_configs)]
        inst = pretty_midi.Instrument(program=cfg["program"], is_drum=cfg["is_drum"], name=cfg["name"])
        roll = np.clip(roll, 0, 1)

        sparse = np.zeros_like(roll)

        # 控制节奏点位
        for t in range(cfg["shift"], time_steps, cfg["beat"]):
            top = np.argsort(roll[:, t])[-cfg["notes"]:]
            sparse[top, t] = 1
        roll = sparse.astype(np.uint8)

        # 转换为 MIDI notes
        for pitch in range(roll.shape[0]):
            note_on = None
            for t in range(roll.shape[1]):
                if roll[pitch, t] == 1 and note_on is None:
                    note_on = t
                elif roll[pitch, t] == 0 and note_on is not None:
                    inst.notes.append(pretty_midi.Note(velocity=100, pitch=pitch,
                                                       start=note_on/fs, end=t/fs))
                    note_on = None
            if note_on is not None:
                inst.notes.append(pretty_midi.Note(velocity=100, pitch=pitch,
                                                   start=note_on/fs, end=time_steps/fs))

        midi.instruments.append(inst)

    return midi





# ==== 主函数 ====
def generate_and_save_music(model_path, latent_dim=100, output_shape=(4, 128, 500), fs=100, save_path="generated_music.mid"):
    generator = Generator(latent_dim, output_shape)
    generator.load_state_dict(torch.load(model_path, map_location="cpu"))
    generator.eval()

    z = torch.randn(1, latent_dim)
    with torch.no_grad():
        piano_roll = generator(z)[0].numpy()

    midi = piano_roll_to_midi(piano_roll, fs=fs)
    midi.write(save_path)
    print(f"🎵 已生成多音色 MIDI 文件：{save_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "fixed_midi", "models", "generator_version2.pth")
    generate_and_save_music(model_path)
