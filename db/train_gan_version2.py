import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
                    note = pretty_midi.Note(velocity=100, pitch=pitch, start=start/fs, end=end/fs)
                    instrument.notes.append(note)
                    active = False
            if active:
                note = pretty_midi.Note(velocity=100, pitch=pitch, start=start/fs, end=roll.shape[1]/fs)
                instrument.notes.append(note)
        midi.instruments.append(instrument)
    midi.write(filename)

# -----------------------------
# 数据预处理部分（多轨 piano roll 表示）
# -----------------------------

def find_all_midi_files(root_dir):
    """
    递归遍历 root_dir 及其所有子目录，返回所有扩展名为 .mid 或 .midi 的文件路径列表。
    """
    midi_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(('.mid', '.midi')):
                midi_files.append(os.path.join(dirpath, filename))
    return midi_files

# 设置 MIDI 数据目录为 db/clean_midi（包含子文件夹）
base_dir = os.path.dirname(__file__)
midi_dir = os.path.join(base_dir,"fixed_midi")
print("MIDI 数据目录：", midi_dir)

all_midi_files = find_all_midi_files(midi_dir)
print("找到的 MIDI 文件数：", len(all_midi_files))
for f in all_midi_files:
    print(f)


def midi_to_multi_piano_roll(midi_file, fs=100, max_tracks=4, fixed_length=500):
    """
    将 MIDI 文件转换为多轨 piano roll 表示，返回 shape (max_tracks, 128, fixed_length)
    仅提取前 max_tracks 个轨道，若轨道不足则补零
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file)
        tracks = []
        for instrument in midi_data.instruments:
            # 如果乐器为空（silent）或属于 percussion，可根据需要筛选，此处直接处理
            roll = instrument.get_piano_roll(fs=fs)
            roll = roll / 127.0
            tracks.append(roll)
            if len(tracks) >= max_tracks:
                break
        # 如果轨道不足，则补充全零矩阵
        if len(tracks) < max_tracks:
            max_time = max((r.shape[1] for r in tracks), default=0)
            for _ in range(max_tracks - len(tracks)):
                tracks.append(np.zeros((128, max_time)))
        # 取所有轨道中最大的时间步数或固定长度
        target_length = fixed_length
        processed = []
        for roll in tracks:
            if roll.shape[1] < target_length:
                pad_width = target_length - roll.shape[1]
                roll = np.pad(roll, ((0, 0), (0, pad_width)), mode='constant')
            else:
                roll = roll[:, :target_length]
            processed.append(roll)
        multi_roll = np.stack(processed, axis=0)  # shape: (max_tracks, 128, target_length)
        return multi_roll
    except Exception as e:
        print(f"Error processing {midi_file}: {e}")
        return None


class MidiDatasetMulti(Dataset):
    def __init__(self, midi_dir, fs=100, fixed_length=500, max_tracks=4):
        self.midi_files = find_all_midi_files(midi_dir)
        self.fs = fs
        self.fixed_length = fixed_length
        self.max_tracks = max_tracks
        self.data = []
        self._prepare_dataset()

    def _prepare_dataset(self):
        for midi_file in self.midi_files:
            multi_roll = midi_to_multi_piano_roll(midi_file, fs=self.fs,
                                                  max_tracks=self.max_tracks,
                                                  fixed_length=self.fixed_length)
            if multi_roll is None:
                continue
            self.data.append(multi_roll)
        if len(self.data) > 0:
            self.data = np.array(self.data)
        else:
            print("未找到有效的 MIDI 数据！")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]  # shape: (max_tracks, 128, fixed_length)
        return torch.tensor(sample, dtype=torch.float32)


# -----------------------------
# GAN 模型（生成多轨 piano roll）
# -----------------------------
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


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.input_shape = input_shape
        self.model = nn.Sequential(
            nn.Linear(np.prod(input_shape), 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),  # 添加 Dropout 防止 D 过拟合
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        validity = self.model(x)
        return validity


# -----------------------------
# 训练函数
# -----------------------------
def train_musegan(midi_dir, epochs=100, batch_size=16, latent_dim=100, fs=100, fixed_length=500, max_tracks=4
     #调用已有model
                  ,
    resume=True,
    generator_path="path/to/generator.pth",
    discriminator_path="path/to/discriminator.pth"):
    dataset = MidiDatasetMulti(midi_dir, fs=fs, fixed_length=fixed_length, max_tracks=max_tracks)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    if len(dataset) == 0:
        print("数据集为空，请检查 MIDI 数据路径。")
        return

    sample_shape = dataset.data[0].shape
    print("样本数据形状:", sample_shape)

    generator = Generator(latent_dim, sample_shape)
    discriminator = Discriminator(sample_shape)

    # 如果启用续训（resume），则加载已有模型参数
    if resume:
        if generator_path and os.path.exists(generator_path):
            generator.load_state_dict(torch.load(generator_path))
            print(f"✅ Loaded Generator from: {generator_path}")
        else:
            print("⚠️ Generator checkpoint not found at", generator_path)

        if discriminator_path and os.path.exists(discriminator_path):
            discriminator.load_state_dict(torch.load(discriminator_path))
            print(f"✅ Loaded Discriminator from: {discriminator_path}")
        else:
            print("⚠️ Discriminator checkpoint not found at", discriminator_path)

    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for i, real_data in enumerate(dataloader):
            current_bs = real_data.size(0)

            # 使用 label smoothing
            valid = torch.ones(current_bs, 1) * 0.9
            fake = torch.zeros(current_bs, 1)

            # ---------------------
            #  训练判别器 Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            validity_real = discriminator(real_data)
            d_loss_real = adversarial_loss(validity_real, valid)

            z = torch.randn(current_bs, latent_dim)
            gen_data = generator(z).detach() + 0.05 * torch.randn_like(generator(z))  # 生成器输出加噪声
            validity_fake = discriminator(gen_data)
            d_loss_fake = adversarial_loss(validity_fake, fake)

            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()

            # ---------------------
            #  多次训练 Generator
            # ---------------------
            for _ in range(2):
                optimizer_G.zero_grad()
                z = torch.randn(current_bs, latent_dim)
                gen_data = generator(z)
                validity_fake = discriminator(gen_data)
                g_loss = adversarial_loss(validity_fake, valid)
                g_loss.backward()
                optimizer_G.step()

            if i % 10 == 0:
                print(
                    f"[Epoch {epoch + 1}/{epochs}] [Batch {i}/{len(dataloader)}] D: {d_loss.item():.4f}  G: {g_loss.item():.4f}")

        # <<< Add here
        # if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            z_sample = torch.randn(1, latent_dim)
            gen_sample = generator(z_sample).squeeze(0).cpu().numpy()
            binarized = (gen_sample > 0.3).astype(np.uint8)
            output_dir = os.path.join(os.path.dirname(__file__), "generated_midis")
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"epoch{epoch + 1}.mid")
            save_pianoroll_as_midi(binarized, save_path)
            print(f"🎵 Saved generated MIDI at epoch {epoch + 1} -> {save_path}")

    models_dir = os.path.join(midi_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    torch.save(generator.state_dict(), os.path.join(models_dir, "generator_version2_2.pth"))
    torch.save(discriminator.state_dict(), os.path.join(models_dir, "discriminator_version2_2.pth"))
    print("模型训练完成，已保存。")



# -----------------------------
# 主函数入口
# -----------------------------
# #创建新model
# if __name__ == "__main__":
#     # 假设 MIDI 数据存放在项目根目录下的 db 目录中
#     base_dir = os.path.dirname(__file__)
#     midi_dir = os.path.join(base_dir,"fixed_midi")
#     # 调用训练函数
#     #train_musegan(midi_dir, epochs=50, batch_size=16, latent_dim=100, fs=100, fixed_length=500, max_tracks=4)


#调用已有model
if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    midi_dir = os.path.join(base_dir, "fixed_midi")
    model_dir = os.path.join(midi_dir, "models")

    generator_ckpt = os.path.join(model_dir, "generator_version2.pth")
    discriminator_ckpt = os.path.join(model_dir, "discriminator_version2.pth")

    train_musegan(
        midi_dir=midi_dir,
        epochs=20,  # 再训练 20 轮
        batch_size=16,
        latent_dim=100,
        fs=100,
        fixed_length=500,
        max_tracks=4,
        resume=True,  # ✅ 这里必须显式设置
        generator_path=generator_ckpt,
        discriminator_path=discriminator_ckpt
    )
