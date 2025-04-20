import torch
import numpy as np
import os
import pretty_midi


def save_pianoroll_as_midi(piano_roll, filename, fs=100):
    midi = pretty_midi.PrettyMIDI()
    for i, roll in enumerate(piano_roll):
        instrument = pretty_midi.Instrument(program=0)
        for pitch in range(128):
            active = False
            start = 0
            for t in range(roll.shape[1]):
                if roll[pitch, t] > 0 and not active:
                    active = True
                    start = t
                elif roll[pitch, t] == 0 and active:
                    end = t
                    note = pretty_midi.Note(velocity=100, pitch=pitch, start=start / fs, end=end / fs)
                    instrument.notes.append(note)
                    active = False
            if active:
                note = pretty_midi.Note(velocity=100, pitch=pitch, start=start / fs, end=roll.shape[1] / fs)
                instrument.notes.append(note)
        midi.instruments.append(instrument)
    midi.write(filename)


# Version 2 Generator: Fully connected
class GeneratorV2(torch.nn.Module):
    def __init__(self, latent_dim, output_shape):
        super(GeneratorV2, self).__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, np.prod(output_shape)),
            torch.nn.Tanh()
        )

    def forward(self, z):
        out = self.fc(z)
        return out.view(z.size(0), *self.output_shape)


def generate_from_trained_model():
    latent_dim = 100
    output_shape = (4, 128, 500)
    model_path = "fixed_midi/models/generator_musegan.pth"

    generator = GeneratorV2(latent_dim, output_shape)
    generator.load_state_dict(torch.load(model_path, map_location='cpu'))
    generator.eval()

    with torch.no_grad():
        z = torch.randn(1, latent_dim)
        gen_sample = generator(z).squeeze(0).cpu().numpy()
        binarized = (gen_sample > 0.3).astype(np.uint8)

        output_dir = os.path.join(os.path.dirname(__file__), "generated_midis")
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "sample_from_version2.mid")
        save_pianoroll_as_midi(binarized, save_path)
        print(f"✅ MIDI file saved to {save_path}")


if __name__ == "__main__":
    generate_from_trained_model()