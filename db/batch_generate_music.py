import os
import torch
import numpy as np
import pretty_midi
from model import Generator
from utils import analyze_music


def piano_roll_to_midi(piano_roll, fs=100):
    midi = pretty_midi.PrettyMIDI()

    instrument_configs = [
        {"program": 73, "is_drum": False, "name": "Flute"},
        {"program": 40, "is_drum": False, "name": "Violin"},
        {"program": 33, "is_drum": False, "name": "Bass"},
        {"program": 0, "is_drum": True,  "name": "Drums"}
    ]

    for i, roll in enumerate(piano_roll):
        config = instrument_configs[i % len(instrument_configs)]
        instrument = pretty_midi.Instrument(program=config["program"], is_drum=config["is_drum"], name=config["name"])

        roll = (roll > 0.1).astype(np.uint8)  # 更宽松的阈值

        for pitch in range(roll.shape[0]):
            note_on = None
            for t in range(roll.shape[1]):
                if roll[pitch, t] and note_on is None:
                    note_on = t
                elif not roll[pitch, t] and note_on is not None:
                    start = note_on / fs
                    end = t / fs
                    instrument.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end))
                    note_on = None
            if note_on is not None:
                instrument.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=note_on/fs, end=roll.shape[1]/fs))

        midi.instruments.append(instrument)

    return midi


def batch_generate_music(model_path, latent_dim=100, output_shape=(4, 128, 1000), count=10, output_dir="generated_batch"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator(latent_dim, output_shape[0], output_shape[1], output_shape[2]).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()

    for i in range(count):
        z = torch.randn(1, latent_dim).to(device)
        with torch.no_grad():
            piano_roll = torch.sigmoid(generator(z))[0].cpu().numpy()

        print(f"\n🎵 Sample {i+1}")
        analyze_music(piano_roll)

        midi = piano_roll_to_midi(piano_roll)
        save_path = os.path.join(output_dir, f"sample_{i+1}.mid")
        midi.write(save_path)
        print(f"✅ Saved: {save_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "fixed_midi", "models", "generator_rhythm.pth")
    batch_generate_music(model_path, count=10)
