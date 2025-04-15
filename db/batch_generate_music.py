import os
import torch
import numpy as np
import pretty_midi
from model import Generator
from utils import analyze_music

def piano_roll_to_midi(piano_roll, fs=100):
    import numpy as np
    import pretty_midi

    midi = pretty_midi.PrettyMIDI()

    # 每个通道指定音域范围（pitch_min, pitch_max）
    channel_pitch_ranges = [
        (72, 84),  # Flute: 明亮高音
        (60, 72),  # Violin: 中高音
        (36, 48),  # Bass: 低音
        (35, 81)   # Drums: GM打击乐
    ]

    instrument_configs = [
        {"program": 73, "is_drum": False, "name": "Flute"},
        {"program": 40, "is_drum": False, "name": "Violin"},
        {"program": 33, "is_drum": False, "name": "Bass"},
        {"program": 0,  "is_drum": True,  "name": "Drums"}
    ]

    def safe_velocity(v):
        return int(np.clip(v, 1, 127))

    def safe_pitch(p):
        return int(np.clip(p, 0, 127))

    for i, roll in enumerate(piano_roll):
        config = instrument_configs[i % len(instrument_configs)]
        pitch_min, pitch_max = channel_pitch_ranges[i]

        instrument = pretty_midi.Instrument(
            program=config["program"],
            is_drum=config["is_drum"],
            name=config["name"]
        )

        bin_roll = (roll > 0.5).astype(np.uint8)

        for pitch in range(bin_roll.shape[0]):
            note_on = None
            for t in range(bin_roll.shape[1]):
                if bin_roll[pitch, t] and note_on is None:
                    note_on = t
                elif not bin_roll[pitch, t] and note_on is not None:
                    start = note_on / fs
                    end = t / fs
                    # 映射 pitch 到乐器范围
                    real_pitch = safe_pitch(int(np.interp(pitch, [0, bin_roll.shape[0] - 1], [pitch_min, pitch_max])))
                    length = t - note_on
                    velocity = safe_velocity(length * 10)

                    instrument.notes.append(pretty_midi.Note(
                        velocity=velocity,
                        pitch=real_pitch,
                        start=start,
                        end=end
                    ))
                    print(f"Channel {i} ({config['name']}), Pitch: {real_pitch}, Velocity: {velocity}, Time: {start:.2f}–{end:.2f}")
                    note_on = None

            if note_on is not None:
                real_pitch = safe_pitch(int(np.interp(pitch, [0, bin_roll.shape[0] - 1], [pitch_min, pitch_max])))
                length = bin_roll.shape[1] - note_on
                velocity = safe_velocity(length * 10)

                instrument.notes.append(pretty_midi.Note(
                    velocity=velocity,
                    pitch=real_pitch,
                    start=note_on / fs,
                    end=bin_roll.shape[1] / fs
                ))
                print(f"Channel {i} ({config['name']}), Pitch: {real_pitch}, Velocity: {velocity}, Time: {note_on/fs:.2f}–{bin_roll.shape[1]/fs:.2f}")

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
    model_path = os.path.join(base_dir, "fixed_midi", "models", "generator_version4.pth")
    batch_generate_music(model_path, count=10)
