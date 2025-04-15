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

# ==== Piano roll 转 MIDI ====
def piano_roll_to_midi(piano_roll, fs=100):
    """
    piano_roll: shape (tracks, 128, time)
    """
    midi = pretty_midi.PrettyMIDI()
    time_steps = piano_roll.shape[2] / fs

    # 音色映射：4轨分别为长笛、小提琴、电贝斯、鼓
    instrument_configs = [
        {"program": 73, "is_drum": False, "name": "Melody_Flute"},
        {"program": 40, "is_drum": False, "name": "Strings_Violin"},
        {"program": 33, "is_drum": False, "name": "Bass_Electric"},
        {"program": 0,  "is_drum": True,  "name": "Drums"}
    ]

    for i, roll in enumerate(piano_roll):
        config = instrument_configs[i % len(instrument_configs)]
        instrument = pretty_midi.Instrument(program=config["program"], is_drum=config["is_drum"], name=config["name"])
        roll = (roll > 0.5).astype(np.uint8)

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


# ==== 主函数 ====
def generate_and_save_music(model_path, latent_dim=100, output_shape=(4, 128, 1000), fs=100, save_path="generated_music.mid"):
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
    model_path = os.path.join(base_dir, "fixed_midi", "models", "generator_version4.pth")
    generate_and_save_music(model_path)
