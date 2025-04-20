import os
import torch
import numpy as np
import pretty_midi
from model import Generator
from utils import reward_from_rhythm


def piano_roll_to_midi(piano_roll, fs=100):
    """
    Convert (tracks, 128, time) piano roll to PrettyMIDI object
    """
    midi = pretty_midi.PrettyMIDI()

    instrument_configs = [
        {"program": 73, "is_drum": False, "name": "Flute"},
        {"program": 40, "is_drum": False, "name": "Violin"},
        {"program": 33, "is_drum": False, "name": "Bass"},
        {"program": 0,  "is_drum": True,  "name": "Drums"}
    ]

    for i, roll in enumerate(piano_roll):
        config = instrument_configs[i % len(instrument_configs)]
        instrument = pretty_midi.Instrument(
            program=config["program"],
            is_drum=config["is_drum"],
            name=config["name"]
        )

        # 更宽松的二值化阈值
        roll = (roll > 0.2).astype(np.uint8)

        has_notes = False
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
                    has_notes = True
                    note_on = None
            if note_on is not None:
                note = pretty_midi.Note(velocity, pitch, note_on / fs, roll.shape[1] / fs)
                instrument.notes.append(note)
                has_notes = True

        if not has_notes:
            print(f"⚠️ Warning: Track {i} ({config['name']}) is empty.")
        midi.instruments.append(instrument)

    return midi


def generate_music(model_path, latent_dim=100, output_shape=(4, 128, 500), save_path="generated_version2.mid"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 载入模型
    generator = Generator(latent_dim, output_shape[0], output_shape[1], output_shape[2]).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()

    # 生成 latent 向量 → 生成 piano roll
    z = torch.randn(1, latent_dim).to(device)
    with torch.no_grad():
        raw_output = generator(z)
        piano_roll = torch.sigmoid(raw_output)[0].cpu().numpy()  # 激活处理

    # 输出统计信息
    print("🎹 Piano roll stats → max:", piano_roll.max(), "min:", piano_roll.min(), "mean:", piano_roll.mean())

    # 节奏奖励评估
    rhythm_score = reward_from_rhythm(piano_roll)
    print(f"🥁 Rhythm Score: {rhythm_score:.2f}  → {'✅ 有节奏!' if rhythm_score > 0 else '⚠️ 节奏较弱'}")

    # 转换并保存为 MIDI
    midi = piano_roll_to_midi(piano_roll)
    midi.write(save_path)
    print(f"🎶 已保存生成音乐文件：{save_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "fixed_midi", "models", "generator_version2.pth")
    generate_music(model_path)
