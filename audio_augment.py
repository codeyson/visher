import torch
import torchaudio
import random


class AudioAugmentor:
    def __init__(self, sample_rate=24000):
        self.sample_rate = sample_rate

    # ── Basic augmentations ────────────────────────────────────────────────────

    def add_noise(self, waveform, noise_level=None):
        """Additive Gaussian noise to simulate mic/electronics noise."""
        if noise_level is None:
            noise_level = random.uniform(0.001, 0.006)
        noise = torch.randn_like(waveform)
        return waveform + noise_level * noise

    def random_gain(self, waveform):
        """Random volume scaling to simulate inconsistent mic gain."""
        gain = random.uniform(0.6, 1.4)
        return waveform * gain

    def clipping(self, waveform):
        """Hard clipping to simulate cheap ADC or overdriven mic input."""
        threshold = random.uniform(0.6, 0.9)
        return torch.clamp(waveform, -threshold, threshold)

    # ── Telecoms / codec augmentations ────────────────────────────────────────

    def resample_distortion(self, waveform):
        """
        Simulate system resampling artifacts (e.g. 24k → 16k → 24k).
        Critical for live detection: most telephony and VoIP pipelines
        internally resample to 8k or 16k.
        """
        target_sr = random.choice([8000, 16000])
        waveform = torchaudio.functional.resample(waveform, self.sample_rate, target_sr)
        waveform = torchaudio.functional.resample(waveform, target_sr, self.sample_rate)
        return waveform

    def compression(self, waveform):
        """
        Soft-knee tanh compression to simulate Zoom/Discord/WebRTC codecs.
        """
        drive = random.uniform(1.5, 3.0)
        return torch.tanh(waveform * drive) / torch.tanh(torch.tensor(drive))

    def bandpass_filter(self, waveform):
        """
        Simulate phone/headset frequency response:
        low cut ~200-400 Hz, high cut ~6k-8k Hz.
        Cheap mics and telephony both roll off the extremes.
        """
        low_cutoff  = random.uniform(200,  400)
        high_cutoff = random.uniform(6000, 8000)

        # Normalise to [0, 1] range for torchaudio
        low_norm  = low_cutoff  / (self.sample_rate / 2)
        high_norm = high_cutoff / (self.sample_rate / 2)

        low_norm  = max(0.001, min(low_norm,  0.999))
        high_norm = max(0.001, min(high_norm, 0.999))
        if low_norm >= high_norm:
            high_norm = low_norm + 0.01

        waveform = torchaudio.functional.highpass_biquad(
            waveform, self.sample_rate, cutoff_freq=low_cutoff
        )
        waveform = torchaudio.functional.lowpass_biquad(
            waveform, self.sample_rate, cutoff_freq=high_cutoff
        )
        return waveform

    def packet_loss(self, waveform):
        """
        Simulate VoIP packet loss: randomly zero-out short chunks (5–30 ms).
        Creates dropout silences not present in clean studio recordings.
        """
        chunk_ms   = random.randint(5, 30)
        chunk_size = int(self.sample_rate * chunk_ms / 1000)
        num_losses = random.randint(1, 3)
        result     = waveform.clone()
        T          = waveform.shape[-1]
        for _ in range(num_losses):
            if T <= chunk_size:
                break
            start = random.randint(0, T - chunk_size)
            result[..., start:start + chunk_size] = 0.0
        return result

    # ── Room / environment augmentations ──────────────────────────────────────

    def add_reverb(self, waveform):
        """
        Simulate room acoustics via a synthetic exponential-decay RIR.
        The single biggest domain gap between clean training data and live mic.
        Decay range covers anechoic (0.05 s) to a small room (~0.5 s).
        """
        decay      = random.uniform(0.05, 0.5)
        impulse_len = min(int(self.sample_rate * decay), self.sample_rate)
        t   = torch.linspace(0, decay, impulse_len, device=waveform.device)
        rir = torch.exp(-6.9 * t / decay) * torch.randn(impulse_len, device=waveform.device)
        rir = rir / (rir.norm() + 1e-8)

        convolved = torchaudio.functional.fftconvolve(waveform, rir.unsqueeze(0))
        return convolved[..., :waveform.shape[-1]]

    def add_background_noise(self, waveform):
        """
        Overlay synthetic stationary background noise (keyboard, HVAC, crowd)
        at a random SNR between 10–30 dB.
        """
        snr_db      = random.uniform(10, 30)
        signal_rms  = waveform.norm() / (waveform.numel() ** 0.5 + 1e-8)
        noise_rms   = signal_rms / (10 ** (snr_db / 20))

        noise_type  = random.choice(['white', 'pink'])
        noise       = torch.randn_like(waveform)

        if noise_type == 'pink':
            # Approximate pink noise by low-pass filtering white noise
            noise = torchaudio.functional.lowpass_biquad(
                noise, self.sample_rate, cutoff_freq=2000
            )

        noise = noise / (noise.norm() / (noise.numel() ** 0.5) + 1e-8) * noise_rms
        return waveform + noise

    # ── Timing augmentations ──────────────────────────────────────────────────

    def time_shift(self, waveform):
        """
        Shift waveform left or right with zero-padding (not roll-wrap).
        roll() creates a discontinuity that doesn't exist in real audio.
        """
        max_shift = int(0.1 * waveform.shape[-1])
        shift     = random.randint(-max_shift, max_shift)
        result    = torch.zeros_like(waveform)
        if shift > 0:
            result[..., shift:] = waveform[..., :-shift]
        elif shift < 0:
            result[..., :shift] = waveform[..., -shift:]
        else:
            result = waveform.clone()
        return result

    def pitch_shift(self, waveform):
        """
        Simulate minor recording chain speed variation via resampling
        (±2 semitones). Cheap proxy for true pitch shifting.
        """
        semitones = random.uniform(-2, 2)
        factor    = 2 ** (semitones / 12)
        orig_len  = waveform.shape[-1]
        new_sr    = int(self.sample_rate * factor)
        if new_sr == self.sample_rate:
            return waveform
        waveform  = torchaudio.functional.resample(waveform, self.sample_rate, new_sr)
        # Trim or pad back to original length
        if waveform.shape[-1] >= orig_len:
            waveform = waveform[..., :orig_len]
        else:
            pad = orig_len - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        return waveform

    # ── Entry point ───────────────────────────────────────────────────────────

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply a random subset of augmentations.

        Args:
            waveform: Tensor of shape (T,) or (1, T)

        Returns:
            Tensor of shape (T,)  — always squeezed back to 1-D
        """
        squeezed = waveform.dim() == 1
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)   # → (1, T)

        # ── Core degradations (higher probability) ─────────────────────────
        if random.random() < 0.5:
            waveform = self.add_noise(waveform)

        if random.random() < 0.5:
            waveform = self.random_gain(waveform)

        if random.random() < 0.4:
            waveform = self.add_background_noise(waveform)

        # ── Room acoustics ─────────────────────────────────────────────────
        if random.random() < 0.5:
            waveform = self.add_reverb(waveform)

        # ── Telecoms / codec chain ─────────────────────────────────────────
        if random.random() < 0.4:
            waveform = self.resample_distortion(waveform)

        if random.random() < 0.35:
            waveform = self.bandpass_filter(waveform)

        if random.random() < 0.3:
            waveform = self.compression(waveform)

        if random.random() < 0.25:
            waveform = self.packet_loss(waveform)

        # ── Timing / dynamics ──────────────────────────────────────────────
        if random.random() < 0.3:
            waveform = self.clipping(waveform)

        if random.random() < 0.3:
            waveform = self.time_shift(waveform)

        if random.random() < 0.2:
            waveform = self.pitch_shift(waveform)

        # ── Always return (T,) to match main.py expectations ──────────────
        return waveform.squeeze(0)