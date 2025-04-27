import os
import torch
import numpy as np
import pretty_midi
from torch import nn

# ==== Generator 定义（保持不变）====
class Generator(nn.Module):
    def __init__(self, latent_dim, output_shape):
        """
        output_shape: (channels, 128, fixed_length)
        """
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

    # 乐器配置（program + 是否是打击乐）

    # 21
    # instrument_configs = [
    #     {"program": 82, "is_drum": False, "name": "Calliope Lead"},  # 主旋律哨音
    #     {"program": 81, "is_drum": False, "name": "Saw Lead"},  # 和声滑音
    #     {"program": 34, "is_drum": False, "name": "Picked Bass"},  # 清晰低音
    #     {"program": 9, "is_drum": True, "name": "Standard Drums"}  # 标准鼓组
    # ]
    #
    # rhythm_patterns = [
    #     list(range(0, 500, 12)),  # Lead: 频密打点
    #     list(range(4, 500, 16)),  # Saw: 反拍或轻节奏
    #     list(range(0, 500, 32)),  # Bass: 拍内节奏线
    #     [0, 8, 16, 24, 32, 40]  # Drums: 稳定节奏骨架
    # ]

    # 20
    # instrument_configs = [
    #     {"program": 85, "is_drum": False, "name": "Vox Lead"},  # 合成器人声旋律
    #     {"program": 91, "is_drum": False, "name": "Pad Bowed"},  # 弓弦氛围感
    #     {"program": 38, "is_drum": False, "name": "Synth Bass 1"},  # 轻低频
    #     {"program": 9, "is_drum": True, "name": "Brush Drums"}  # 轻打节奏
    # ]
    #
    # rhythm_patterns = [
    #     list(range(0, 500, 24)),  # Vox Lead: 松节奏旋律
    #     list(range(16, 500, 48)),  # Pad: 拉长背景
    #     list(range(4, 500, 32)),  # Bass: 简洁低频
    #     [0, 12, 24, 36, 60]  # Drums: 偏 soft 节奏点
    # ]

    # 19
    # instrument_configs = [
    #     {"program": 80, "is_drum": False, "name": "Square Lead"},  # 方波旋律
    #     {"program": 90, "is_drum": False, "name": "Polysynth Pad"},  # 合成铺底
    #     {"program": 39, "is_drum": False, "name": "Synth Bass 2"},  # 粗颗粒低音
    #     {"program": 9, "is_drum": True, "name": "Power Drums"}  # 动作感鼓组
    # ]
    #
    # rhythm_patterns = [
    #     list(range(0, 500, 16)),  # Lead: 主旋律均匀拍
    #     list(range(8, 500, 32)),  # Pad: 偏慢拍落点
    #     list(range(0, 500, 24)),  # Bass: 加入律动细分
    #     [0, 8, 16, 24, 40, 48]  # Drums: 强节奏骨架
    # ]


    # 18
    # instrument_configs = [
    #     {"program": 8, "is_drum": False, "name": "Celesta"},  # ✨ 闪烁高音
    #     {"program": 95, "is_drum": False, "name": "Pad Sweep"},  # 🌫️ 空间氛围
    #     {"program": 50, "is_drum": False, "name": "Slow Strings"},  # 🎻 慢弦和声
    #     {"program": 9, "is_drum": True, "name": "Brush Drums"}  # 🥁 柔鼓刷击
    # ]
    #
    # rhythm_patterns = [
    #     list(range(0, 500, 24)),  # Celesta 星星点点
    #     list(range(12, 500, 64)),  # Pad 低频起伏
    #     list(range(6, 500, 48)),  # Strings 旋律延展
    #     [0, 12, 24, 36]  # Drum 轻节奏铺垫
    # ]

    # 17
    # instrument_configs = [
    #     {"program": 46, "is_drum": False, "name": "Harp"},  # 🎼 点缀旋律
    #     {"program": 54, "is_drum": False, "name": "Synth Voice"},  # 🧘 人声pad氛围
    #     {"program": 35, "is_drum": False, "name": "Fretless Bass"},  # 🎸 滑音低频
    #     {"program": 9, "is_drum": True, "name": "Standard Drums"}  # 🥁 标准鼓节奏
    # ]
    #
    # rhythm_patterns = [
    #     list(range(0, 500, 12)),  # Harp 每小节内轮指风
    #     list(range(16, 500, 48)),  # Voice Pad：每段落落点
    #     list(range(4, 500, 32)),  # Bass 偏稳的切分
    #     [0, 8, 16, 24, 40]  # Drum 带律动基础鼓点
    # ]

    # 16
    # instrument_configs = [
    #     {"program": 5, "is_drum": False, "name": "Electric Piano"},  # 🎹 轻快主旋律
    #     {"program": 89, "is_drum": False, "name": "Warm Pad"},  # 🌫️ 背景铺底
    #     {"program": 32, "is_drum": False, "name": "Acoustic Bass"},  # 🎸 柔和低音
    #     {"program": 9, "is_drum": True, "name": "Room Drums"}  # 🥁 鼓，空间感
    # ]
    #
    # rhythm_patterns = [
    #     list(range(0, 500, 16)),  # Electric Piano 每拍落点
    #     list(range(8, 500, 32)),  # Pad 稀疏落点
    #     list(range(0, 500, 24)),  # Bass 三连节奏点
    #     [0, 8, 16, 24, 48, 56]  # Drum 基础节奏组合
    # ]

    # 像素15
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
        [0, 8, 16, 24, 32, 40]  # 🥁 Drum: kick/snare 组合
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
def generate_and_save_music(model_path, latent_dim=100, output_shape=(4, 128, 500), fs=100,
                            save_path="generated_music.mid"):
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