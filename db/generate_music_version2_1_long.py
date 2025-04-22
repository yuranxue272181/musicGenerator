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
    幻神终极清音版：
    - 限制 pitch 范围（避免尖叫）
    - 调整 velocity（柔和输出）
    - 使用节奏模板控制落点
    - 鼓轨修复为标准 GM Percussion
    """
    import pretty_midi
    midi = pretty_midi.PrettyMIDI()
    time_steps = piano_roll.shape[2]

#像素15
    instrument_configs = [
        {"program": 80, "is_drum": False, "name": "Square Lead"},  # 方波旋律
        {"program": 81, "is_drum": False, "name": "Saw Lead"},  # 锯齿和声/点缀
        {"program": 38, "is_drum": False, "name": "Synth Bass"},  # 三角波低音
        {"program": 9, "is_drum": True, "name": "Noise Drums"}  # 噪声鼓组
    ]
    rhythm_patterns = [
        list(range(0, time_steps, 16)),  # 🎹 Piano: 每拍落点
        list(range(4, time_steps, 24)),  # 🎸 Guitar: off-beat
        list(range(0, time_steps, 32)),  # 🎸 Bass: 稀疏根音
        [0, 8, 16, 24, 32, 40]           # 🥁 Drum: kick/snare 组合
    ]

    for i, roll in enumerate(piano_roll):
        cfg = instrument_configs[i % len(instrument_configs)]
        pattern = rhythm_patterns[i % len(rhythm_patterns)]

        # 清理过高音符（尖锐声源）
        roll = np.clip(roll, 0, 1)
        roll[95:, :] = 0  # ✅ 限制 pitch 最大值（安全区域）

        # 只保留节奏位置的最强 note
        sparse = np.zeros_like(roll)
        for t in pattern:
            top = np.argsort(roll[:, t])[-1:]
            sparse[top, t] = 1

        inst = pretty_midi.Instrument(program=cfg["program"], is_drum=cfg["is_drum"], name=cfg["name"])

        # 合成 note，统一使用较低力度
        for pitch in range(128):
            note_on = None
            for t in range(time_steps):
                if sparse[pitch, t] == 1 and note_on is None:
                    note_on = t
                elif sparse[pitch, t] == 0 and note_on is not None:
                    inst.notes.append(pretty_midi.Note(
                        velocity=80, pitch=pitch,  # ✅ 柔和音量
                        start=note_on / fs,
                        end=t / fs
                    ))
                    note_on = None
            if note_on is not None:
                inst.notes.append(pretty_midi.Note(
                    velocity=80, pitch=pitch,
                    start=note_on / fs, end=time_steps / fs
                ))

        midi.instruments.append(inst)

    return midi








# ==== 主函数 ====

# #生成四段不一样的并组合
# def generate_and_save_music(model_path, latent_dim=100, output_shape=(4, 128, 500), fs=100,
#                             save_path="generated_music_long.mid", num_segments=4):
#     generator = Generator(latent_dim, output_shape)
#     generator.load_state_dict(torch.load(model_path, map_location="cpu"))
#     generator.eval()
#
#     # 批量生成 latent 向量
#     z = torch.randn(num_segments, latent_dim)
#
#     piano_rolls = []
#     with torch.no_grad():
#         generated = generator(z).numpy()  # shape: (num_segments, 4, 128, 500)
#         for i in range(num_segments):
#             piano_rolls.append(generated[i])
#
#     # 拼接成一个长的 piano roll，沿时间维度 axis=2
#     long_piano_roll = np.concatenate(piano_rolls, axis=2)  # shape: (4, 128, 500*num_segments)
#
#     midi = piano_roll_to_midi(long_piano_roll, fs=fs)
#     midi.write(save_path)
#     print(f"🎵 已生成组合多段的多音色 MIDI 文件：{save_path}")


#生成一段并重复四次
def generate_and_save_music(model_path, latent_dim=100, output_shape=(4, 128, 500), fs=100,
                            save_path="repeated_music.mid", repeat_times=4):
    generator = Generator(latent_dim, output_shape)
    generator.load_state_dict(torch.load(model_path, map_location="cpu"))
    generator.eval()

    # 只生成一个 latent 向量
    z = torch.randn(1, latent_dim)

    with torch.no_grad():
        piano_roll = generator(z)[0].numpy()  # shape: (4, 128, 500)

    # 沿时间轴重复拼接（axis=2 是时间维度）
    repeated_roll = np.tile(piano_roll, (1, 1, repeat_times))  # shape: (4, 128, 500 * repeat_times)

    midi = piano_roll_to_midi(repeated_roll, fs=fs)
    midi.write(save_path)
    print(f"🎵 已生成重复 {repeat_times} 次的多音色 MIDI 文件：{save_path}")



if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "fixed_midi", "models", "generator_version2.pth")
    generate_and_save_music(model_path)
