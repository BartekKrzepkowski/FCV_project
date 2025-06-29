import os
import math
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchaudio
from typing import List, Optional, Tuple, Union
from pathlib import Path
from tqdm import tqdm

class SpectrogramDatasetGenerator:
    def __init__(
        self,
        df,
        custom_root: Optional[Path] = None,
        sample_rate: int = 16000,
        desired_duration_sec: int = 5,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        eps: float = 1e-9,
        format: str = "pt"
    ):
        self.custom_root = custom_root
        self.sample_rate = sample_rate
        self.desired_duration_sec = desired_duration_sec
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.eps = eps
        self.format = format
        self.df = df

        if isinstance(self.root_data, str):
            self.root_data = Path(self.root_data)
        self._validate_root_directory()

        self.root_spectrograms = self.root_data / "spectrograms_dataset"
        self.root_voices = self.root_data / "speech"

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.hop_length,
            power=2.0
        )

        if self.format not in ["pt"]:
            raise ValueError(f"Unsupported format: {self.format}. Choose 'pt'.")
        
    def _validate_root_directory(self) -> None:
        """Check if root directory exists or can be created."""
        try:
            self.root_data.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize root directory at {self.root_data}: {e}")

    def generate_spectrograms(self) -> None:
        # all_files = self._get_audio_files()
        all_files = [os.path.join(self.root_data.parent, file) for file in self.df['filename'].values if file.endswith('.wav')]

        for file_path in tqdm(all_files, desc="Generating spectrograms"):
            try:
                output_path = self._get_output_path(file_path)
                # Sprawdzenie czy spektrogram już istnieje
                if output_path.exists():
                    continue  # Pomija tworzenie, jeśli już istnieje

                waveform, sample_rate = self._load_audio(file_path)
                if waveform is not None:
                    spectrogram = self._create_spectrogram(waveform)
                    
                    self._save_spectrogram(spectrogram, output_path)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue

    def _get_audio_files(self) -> List[str]:
        audio_files = []
        for root, _, files in os.walk(self.root_voices):
            for file in files:
                if file.endswith('.wav'):
                    audio_files.append(os.path.join(root, file))
        return audio_files

    def _load_audio(self, filepath: str) -> Tuple[Optional[torch.Tensor], Optional[int]]:
        try:
            waveform, sample_rate = torchaudio.load(filepath)

            # 1. Mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # 2. Normalizacja RMS
            waveform = self._normalize_rms(waveform)

            # 3. Przycinanie ciszy (VAD)
            waveform = self._trim_silence(waveform, sample_rate)

            # 4. Resampling
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
                waveform = resampler(waveform)

            num_samples = self.desired_duration_sec * self.sample_rate
            current_len = waveform.shape[1]
            if current_len >= num_samples:
                waveform = waveform[:, :num_samples]
            else:
                # Opcjonalnie: Padding, jeśli dźwięk krótszy niż 5 sek.
                repeats = num_samples // current_len + 1  # liczba powtórzeń potrzebna do osiągnięcia długości
                waveform = waveform.repeat(1, repeats)[:, :num_samples]

            return waveform, self.sample_rate
        except Exception as e:
            print(f"Error loading {filepath}: {str(e)}")
            return None, None

    def _create_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        # # 5. Pre-emphasis
        waveform = torchaudio.functional.preemphasis(waveform, coeff=0.97)

        # 6. Mel-spektrogram i log
        mel_spec = self.mel_spectrogram(waveform)
        return torch.log(mel_spec + self.eps)

    def _normalize_rms(self, waveform: torch.Tensor, target_dbfs: float = -20.0) -> torch.Tensor:
        rms = waveform.pow(2).mean().sqrt()
        target_rms = 10 ** (target_dbfs / 20)
        gain = target_rms / (rms + self.eps)
        return waveform * gain

    def _trim_silence(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        vad = torchaudio.transforms.Vad(sample_rate=sample_rate)
        return vad(waveform)
    
    def _get_output_path(self, input_path: str) -> Path:
        relative_path = Path(input_path).relative_to(self.root_voices)
        stem = relative_path.stem
        output_dir = self.root_spectrograms / relative_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"{stem}.pt"

    def _save_spectrogram(self, spectrogram: torch.Tensor, output_path: Path) -> None:

        torch.save(spectrogram, output_path)
    

    def display_spectrograms(
        self,
        pt_paths: List[Union[str, Path]],
        cols: int = 3,
        figsize: tuple = (14, 4),
        show_freq_axis: bool = True,
    ) -> None:
        """
        Wyświetl siatkę spektrogramów z osiami czasu, częstotliwości i legendą dB.

        Args:
            pt_paths: lista ścieżek do plików .pt
            cols:     liczba kolumn w siatce
            figsize:  rozmiar całej figury (szer., wys.) w calach
            show_freq_axis: jeśli False -> ukrywa oś Y dla oszczędności miejsca
        """
        if len(pt_paths) == 0:
            print("Brak plików do wyświetlenia.")
            return

        # ── parametry konwersji ramek -> sekundy ────────────────────
        seconds_per_frame = self.hop_length / self.sample_rate

        rows = math.ceil(len(pt_paths) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

        for ax, pt_path in zip(axes.flatten(), pt_paths):
            spec = torch.load(pt_path).squeeze(0).cpu().numpy()   # [mels, frames]

            # extent = [xmin, xmax, ymin, ymax] → tu: czas [s], mel-bin
            xmax = spec.shape[1] * seconds_per_frame
            extent = [0, xmax, 0, self.n_mels]

            im = ax.imshow(
                spec,
                origin="lower",
                aspect="auto",
                extent=extent
            )
            ax.set_title(Path(pt_path).stem, fontsize=9)

            # Osie
            ax.set_xlabel("Czas [s]")
            if show_freq_axis:
                ax.set_ylabel("Mel-pasm")
            else:
                ax.set_yticks([])

            # Colorbar (dB)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("Moc [dB]")

        # Ukryj puste osie
        for ax in axes.flatten()[len(pt_paths):]:
            ax.axis("off")

        plt.tight_layout()
        plt.show()


def generate_spectrograms_from_df(
    sample_rate: int = 16000,
    desired_duration_sec: int = 5,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
    eps: float = 1e-9,
    format: str = "pt"
) -> None:
    """
    Funkcja pomocnicza do generowania spektrogramów z DataFrame.
    
    Args:
        sample_rate, desired_duration_sec, n_mels, n_fft, hop_length, eps, format: parametry przetwarzania
    """
    custom_root = '/net/pr2/projects/plgrid/plggdnnp/datasets/VOiCES_devkit/distant-16k'
    df_train = pd.read_csv("data/train_df1.csv")
    df_val = pd.read_csv("data/val_df1.csv")
    df_test = pd.read_csv("data/test_df1.csv")
    df = pd.concat([df_train, df_val, df_test], ignore_index=True)
    generator = SpectrogramDatasetGenerator(
        df,
        custom_root,
        sample_rate=sample_rate,
        desired_duration_sec=desired_duration_sec,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        eps=eps,
        format=format
    )
    generator.generate_spectrograms()