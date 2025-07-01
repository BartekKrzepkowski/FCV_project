import pandas as pd
import torch
import torchvision.transforms as T
import torchaudio.transforms as AT

from src.data.datasets import SpectrogramDataset
import random

def freq_shift(max_bins=2):
    def _fn(spec):
        shift = random.randint(-max_bins, max_bins)
        return torch.roll(spec, shifts=shift, dims=-2)
    return _fn

def db_jitter(low=-2.0, high=2.0):
    def _fn(spec):
        noise = torch.empty_like(spec).uniform_(low, high)
        return spec + noise
    return _fn

def cutout_rect(time_size=10, freq_size=10):
    def _fn(spec):
        t0 = random.randint(0, spec.shape[-1] - time_size - 1)
        f0 = random.randint(0, spec.shape[-2] - freq_size - 1)
        spec[..., f0:f0+freq_size, t0:t0+time_size] = 0.0
        return spec
    return _fn

import random, glob, torch, torchaudio
from pathlib import Path

class AdditiveNoiseSNR:
    def __init__(self, snr_range=(10, 30), noise_dir="musan/noise", sample_rate=16_000):
        self.snr_range   = snr_range
        self.sample_rate = sample_rate
        self.noise_paths = glob.glob(str(Path(noise_dir) / "**/*.wav"), recursive=True)

    def _scale_to_snr(self, clean, noise, snr_db):
        clean_pow = clean.pow(2).mean()
        noise_pow = noise.pow(2).mean() + 1e-9
        desired_noise_pow = clean_pow / (10 ** (snr_db / 10.0))
        return noise * torch.sqrt(desired_noise_pow / noise_pow)

    def __call__(self, waveform):
        if not self.noise_paths:
            return waveform
        noise_path = random.choice(self.noise_paths)
        noise, sr  = torchaudio.load(noise_path)
        if sr != self.sample_rate:
            noise = torchaudio.functional.resample(noise, sr, self.sample_rate)

        if noise.shape[1] < waveform.shape[1]:
            repeats = waveform.shape[1] // noise.shape[1] + 1
            noise = noise.repeat(1, repeats)
        noise = noise[:, : waveform.shape[1]]

        snr_db  = random.uniform(*self.snr_range)
        noise   = self._scale_to_snr(waveform, noise, snr_db)
        return waveform + noise


class ReverbRIR:
    def __init__(self, rir_dir="RIRS_NOISES/real_rir", prob=0.6, sample_rate=16_000):
        self.prob        = prob
        self.sample_rate = sample_rate
        self.rir_paths   = glob.glob(str(Path(rir_dir) / "**/*.wav"), recursive=True)

    def __call__(self, waveform):
        if random.random() > self.prob or not self.rir_paths:
            return waveform

        rir_path = random.choice(self.rir_paths)
        rir, sr  = torchaudio.load(rir_path)
        if sr != self.sample_rate:
            rir = torchaudio.functional.resample(rir, sr, self.sample_rate)

        rir = rir / torch.norm(rir)
        reverbed = torch.nn.functional.conv1d(
            waveform.unsqueeze(0), rir.unsqueeze(0), padding=rir.shape[-1]-1
        ).squeeze(0)
        return reverbed[:, : waveform.shape[1]]


class SpecCrop5s:
    """
    Przytnij lub dopaduj spektrogram do 5 s (≈ 501 ramek).

    Args:
        target_frames (int): liczba ramek po przycięciu; domyślnie 501
        random_start  (bool): True → losowy start; False → od początku
        pad_with      (str): 'repeat'- powtarza ostatnią ramkę,
                             'zeros' - dopaduje zerami
    """
    def __init__(self,
                 target_frames: int = 501,
                 random_start: bool = True,
                 pad_with: str = "repeat",
                 phase: str = "train"  
                 ):
        self.target_frames = target_frames
        self.random_start = random_start
        assert pad_with in ("repeat", "zeros")
        self.pad_with = pad_with
        self.phase = phase

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """
        spec: Tensor [..., mels, frames]  lub  [mels, frames]
        Zwraca tensor o tej samej liczbie wymiarów z frames == 501.
        """
        if spec.dim() == 3:    # [C, mels, frames]
            mels_dim, time_dim = -2, -1
        elif spec.dim() == 2:  # [mels, frames]
            mels_dim, time_dim = -2, -1
        else:
            raise ValueError("Spodziewam się tensoru 2- lub 3-D")

        frames = spec.shape[time_dim]

        if frames >= self.target_frames:
            if self.phase == 'train': 
                max_start = frames - self.target_frames
                start = random.randint(0, max_start) if max_start > 0 else 0
                end = start + self.target_frames
                spec = spec[:,:, start:end] if spec.dim() == 3 else spec[:, start:end]
            else:
                spec = spec[:, :self.target_frames] if spec.dim() == 2 else spec[:,:, :self.target_frames]

        else:
            pad = self.T - frames
            if self.pad_with == "repeat":
                # repetition of the last frame
                last = spec.index_select(time_dim, torch.tensor([frames - 1]))
                repeat = last.repeat_interleave(pad, dim=time_dim)
                spec = torch.cat([spec, repeat], dim=time_dim)
            else:  # zeros
                pad_tensor = torch.zeros_like(spec[..., :pad])
                spec = torch.cat([spec, pad_tensor], dim=time_dim)

        return spec.contiguous() 



def get_voices(custom_root, df_train_path, df_val_path, df_test_path, use_transform=False):
    df_train = pd.read_csv(df_train_path)
    df_val = pd.read_csv(df_val_path)
    df_test = pd.read_csv(df_test_path)
    
    # global_mean, global_std = compute_global_mean_std_from_paths(
    #     df_train['filename'].apply(lambda x: os.path.join(custom_root, x)).tolist()
    # )
    # logging.info(f"Global mean: {global_mean}, Global std: {global_std}")
    # mean = [global_mean.item()]    # lista (per-kanał)
    # std  = [global_std.item()]     # lista (per-kanał)

    mean = [-3.3971]    # lista (per-kanał)
    std  = [2.8988]     # lista (per-kanał)
    normalize = T.Normalize(mean=mean, std=std)

    train_augment = T.Compose([
        SpecCrop5s(phase='train'),  
    # --- quality-enhancement + domain shift ---

        # AdditiveNoiseSNR(snr_range=(10, 30), noise_dir="musan/noise"),
        # ReverbRIR(rir_dir="RIRS_NOISES/real_rir", prob=0.6),

        # --- lekka regularizacja widma ---
        AT.FrequencyMasking(freq_mask_param=12),
        AT.TimeMasking(time_mask_param=20),
        cutout_rect(time_size=10, freq_size=10),

        # # --- istniejące modyfikacje ---
        freq_shift(max_bins=2),
        # db_jitter(low=-2.0, high=2.0),
        normalize
    ])

    val_augment = T.Compose([
        SpecCrop5s(phase='val'),
        normalize
    ])

    train_augment = train_augment if use_transform else None
    val_augment = val_augment if use_transform else None

    train_dataset = SpectrogramDataset(df_train, custom_root=custom_root, transform=train_augment)
    val_dataset = SpectrogramDataset(df_val , custom_root=custom_root, transform=val_augment)
    test_dataset = SpectrogramDataset(df_test, custom_root=custom_root, transform=val_augment)
    return train_dataset, val_dataset, test_dataset


from tqdm import tqdm
from typing import List

def compute_global_mean_std_from_paths(paths: List[str]) -> tuple[torch.Tensor, torch.Tensor]:
    total_sum = 0.0
    total_sumsq = 0.0
    total_count = 0

    for path in tqdm(paths, desc="Analiza spektrogramów"):
        spec = torch.load(path, weights_only=True).float()

        if spec.dim() == 2:
            spec = spec.unsqueeze(0)  # [1, mel, time]

        total_sum   += spec.sum().item()
        total_sumsq += (spec ** 2).sum().item()
        total_count += spec.numel()

    mean = total_sum / total_count
    var  = (total_sumsq / total_count) - mean**2
    std  = var ** 0.5

    return torch.tensor(mean), torch.tensor(std)
