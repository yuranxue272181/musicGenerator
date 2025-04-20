import os
import torch
import numpy as np
import pretty_midi
from torch import nn

# ==== Generator 定义（与训练时保持一致）====
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

# ==== piano roll -> 单轨 piano MIDI ====
def piano_roll_to_piano_midi(roll, fs=100):
    """
    roll: shape (128, time)
    """
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0, is_drum=False, name="Piano")
    roll = (roll > 0.5).astype(np.uint8)  # 二值化

    for pitch in range(roll.shape[0]):
        velocity = 100
        note_on = None
        for t in range(roll.shape[1]):
            if roll[pitch, t] and note_on is None:
                note_on = t
            elif not roll[pitch, t] and note_on is not None:
                start = note_on / fs
                end = t / fs
                note = pretty_midi.Note(velocity, pitch, start, end)
                instrument.notes.append(note)
                note_on = None
        if note_on is not None:
            note = pretty_midi.Note(velocity, pitch, note_on / fs, roll.shape[1] / fs)
            instrument.notes.append(note)

    midi.instruments.append(instrument)
    return midi

# ==== 主函数：只提取第0轨作为piano ====
def generate_and_save_piano_music(model_path, latent_dim=100, output_shape=(4, 128, 500), fs=100, save_path="generated_piano.mid"):
    generator = Generator(latent_dim, output_shape)
    generator.load_state_dict(torch.load(model_path, map_location="cpu"))
    generator.eval()

    z = torch.randn(1, latent_dim)
    with torch.no_grad():
        piano_roll = generator(z)[0][0].numpy()  # ✅ 只取第 0 轨

    print("🎼 Piano roll shape:", piano_roll.shape, "| max:", piano_roll.max(), "| mean:", piano_roll.mean())

    midi = piano_roll_to_piano_midi(piano_roll, fs=fs)
    midi.write(save_path)
    print(f"🎹 单轨钢琴 MIDI 文件已生成：{save_path}")

# ==== 调用示例 ====
if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "fixed_midi", "models", "generator_version2.pth")
    generate_and_save_piano_music(model_path)
