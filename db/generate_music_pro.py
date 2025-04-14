# generate_music.py

import os
import torch
import numpy as np
import pretty_midi
from model import Generator
from utils import reward_from_rhythm


def piano_roll_to_midi(piano_roll, fs=100):
    """
    piano_roll: shape (tracks, 128, time)
    """
    midi = pretty_midi.PrettyMIDI()

    instrument_configs = [
        {"program": 73, "is_drum": False, "name": "Flute"},
        {"program": 40, "is_drum": False, "name": "Violin"},
        {"program": 33, "is_drum": False, "name": "Bass"},
        {"program": 0, "is_drum": True, "name": "Drums"}
    ]

    for i, roll in enumerate(piano_roll):
        config = instrument_configs[i % len(instrument_configs)]
        instrument = pretty_midi.Instrument(program=config["program"], is_drum=config["is_drum"], name=config["name"])

        # 二值化
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


def generate_music(model_path, latent_dim=100, output_shape=(4, 128, 500), save_path="generated_rhythm_music.mid"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator(latent_dim, output_shape[0], output_shape[1], output_shape[2]).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()

    z = torch.randn(1, latent_dim).to(device)
    with torch.no_grad():
        piano_roll = generator(z)[0].cpu().numpy()

    # 评估节奏得分
    rhythm_score = reward_from_rhythm(piano_roll)
    print(f"🥁 Rhythm Score: {rhythm_score:.2f}  → {'有节奏!' if rhythm_score > 0 else '节奏较弱'}")

    midi = piano_roll_to_midi(piano_roll)
    midi.write(save_path)
    print(f"🎶 已生成音乐文件：{save_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "fixed_midi", "models", "generator_rhythm.pth")
    generate_music(model_path)
