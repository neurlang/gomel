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


def pad(audio_buffer, window):
    """
    Apply padding to audio buffer for proper STFT processing.
    
    Args:
        audio_buffer: 1D numpy array of audio samples
        window: STFT window size
        
    Returns:
        Padded audio buffer as numpy array
    """
    min_target_size = 15 * window
    buffer_len = len(audio_buffer)
    
    if buffer_len >= min_target_size:
        # Calculate padding to make length a multiple of window/4
        hop_size = window // 4
        remainder = buffer_len % hop_size
        if remainder != 0:
            pad_amount = hop_size - remainder
            return np.pad(audio_buffer, (0, pad_amount), mode='constant', constant_values=0)
        return audio_buffer
    else:
        # Pad to min_target_size
        pad_amount = min_target_size - buffer_len
        return np.pad(audio_buffer, (0, pad_amount), mode='constant', constant_values=0)


def is_padded(original_length, padded_length, window):
    """
    Detect if audio was padded based on length comparison.
    
    Args:
        original_length: Original audio buffer length
        padded_length: Length after padding
        window: STFT window size
        
    Returns:
        True if audio was padded, False otherwise
    """
    min_target_size = 15 * window
    
    if original_length >= min_target_size:
        hop_size = window // 4
        remainder = original_length % hop_size
        if remainder != 0:
            expected_padded = original_length + (hop_size - remainder)
            return padded_length == expected_padded
        return padded_length == original_length
    else:
        return padded_length == min_target_size


def spectral_normalize(spectrogram):
    """
    Apply spectral normalization using log2 transformation.
    
    Args:
        spectrogram: 2D numpy array of shape (time_frames * num_freqs, 3)
        
    Returns:
        Normalized spectrogram with log2 transformation applied
    """
    epsilon = 1e-10
    # Apply epsilon to prevent log(0), then apply log2 to all 3 channels
    return np.log2(spectrogram + epsilon)


def spectral_denormalize(spectrogram):
    """
    Apply spectral denormalization using exp2 transformation (inverse of normalize).
    
    Args:
        spectrogram: 2D numpy array of shape (time_frames * num_freqs, 3)
        
    Returns:
        Denormalized spectrogram with exp2 transformation applied
    """
    # Apply exp2 to all 3 channels (inverse of log2)
    return np.exp2(spectrogram)


def shrink(spectrogram, resolut, num_freqs):
    """
    Reduce frequency bins from resolut/2 to num_freqs.
    
    Args:
        spectrogram: 2D numpy array of shape (time_frames * resolut/2, 3)
        resolut: FFT resolution
        num_freqs: Target number of frequency bins
        
    Returns:
        Shrunken spectrogram of shape (time_frames * num_freqs, 3)
    """
    original_bins = resolut // 2
    time_frames = len(spectrogram) // original_bins
    
    # Reshape to (time_frames, original_bins, 3)
    reshaped = spectrogram.reshape(time_frames, original_bins, 3)
    
    # Take only the first num_freqs bins
    shrunken = reshaped[:, :num_freqs, :]
    
    # Reshape back to (time_frames * num_freqs, 3)
    return shrunken.reshape(time_frames * num_freqs, 3)


def grow(spectrogram, resolut, num_freqs):
    """
    Expand frequency bins from num_freqs to resolut/2.
    
    Args:
        spectrogram: 2D numpy array of shape (time_frames * num_freqs, 3)
        resolut: FFT resolution
        num_freqs: Current number of frequency bins
        
    Returns:
        Expanded spectrogram of shape (time_frames * resolut/2, 3)
    """
    target_bins = resolut // 2
    time_frames = len(spectrogram) // num_freqs
    
    # Reshape to (time_frames, num_freqs, 3)
    reshaped = spectrogram.reshape(time_frames, num_freqs, 3)
    
    # Create expanded array
    expanded = np.zeros((time_frames, target_bins, 3), dtype=spectrogram.dtype)
    
    # Copy existing bins
    expanded[:, :num_freqs, :] = reshaped
    
    # Replicate last frequency bin to fill expanded space
    if target_bins > num_freqs:
        last_bin = reshaped[:, -1:, :]  # Shape: (time_frames, 1, 3)
        expanded[:, num_freqs:, :] = last_bin
    
    # Reshape back to (time_frames * target_bins, 3)
    return expanded.reshape(time_frames * target_bins, 3)
