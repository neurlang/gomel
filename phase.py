"""
Phase-preserving spectrogram encoder/decoder.

This module implements a phase-preserving spectrogram encoder/decoder that converts
audio waveforms to spectrograms and back to audio without loss of phase information,
enabling high-quality audio reconstruction.
"""

import numpy as np
from scipy import signal
import soundfile as sf
from PIL import Image
import struct


class Phase:
    """Phase-preserving spectrogram encoder/decoder."""
    
    def __init__(self, sample_rate=None, window=1280, resolut=4096, 
                 y_reverse=True, volume_boost=0.0):
        """
        Initialize Phase encoder/decoder.
        
        Args:
            sample_rate: Audio sample rate (determines num_freqs if not provided)
            window: STFT window size (default: 1280)
            resolut: FFT resolution (default: 4096)
            y_reverse: Flip Y-axis in PNG images (default: True)
            volume_boost: Volume multiplier for reconstruction (default: 0.0 = no boost)
        """
        self.sample_rate = sample_rate
        self.window = window
        self.resolut = resolut
        self.y_reverse = y_reverse
        self.volume_boost = volume_boost
        
        # Set num_freqs based on sample rate
        if sample_rate is not None:
            if sample_rate in [8000, 16000, 48000]:
                self.num_freqs = 768
            elif sample_rate in [11025, 22050, 44100]:
                self.num_freqs = 836
            else:
                raise ValueError(
                    f"Unsupported sample rate: {sample_rate}. "
                    f"Supported rates are: 8000, 16000, 48000, 11025, 22050, 44100"
                )
        else:
            raise ValueError(
                f"Unknown sample rate."
                f"Specify sample rate using Phase(sample_rate=)"
            )
