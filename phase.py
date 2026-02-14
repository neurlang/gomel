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
    
    def __init__(self, sample_rate=None, num_freqs=None, window=1280, 
                 resolut=4096, y_reverse=True, volume_boost=0.0):
        """
        Initialize Phase encoder/decoder.
        
        Args:
            sample_rate: Audio sample rate (determines num_freqs if not provided)
            num_freqs: Number of frequency bins (auto-set based on sample_rate if not provided)
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
        
        # Set num_freqs based on sample rate or use provided value
        if num_freqs is not None:
            # Use explicitly provided num_freqs
            self.num_freqs = num_freqs
        elif sample_rate is not None:
            # Determine num_freqs from sample rate
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
            # Default to 768 if neither sample_rate nor num_freqs is provided
            self.num_freqs = 768
    
    def to_phase(self, audio_buffer):
        """
        Convert audio buffer to phase-preserving spectrogram.
        
        Args:
            audio_buffer: 1D numpy array of float64 audio samples
            
        Returns:
            2D numpy array of shape (time_frames * num_freqs, 3) containing
            [realn1, realm0, realm1] for each time-frequency bin
        """
        # Subtask 5.1: Apply padding and compute STFT
        padded_audio = pad(audio_buffer, self.window)
        
        # Compute STFT with Hann window, requesting full two-sided spectrum
        # noverlap = window - hop_size, where hop_size = window // 4
        hop_size = self.window // 4
        noverlap = self.window - hop_size
        
        frequencies, times, stft_result = signal.stft(
            padded_audio,
            window='hann',
            nperseg=self.window,
            noverlap=noverlap,
            nfft=self.resolut,
            return_onesided=False
        )
        
        # Subtask 5.2: Extract 3-channel phase representation from STFT
        # stft_result now has shape (resolut, time_frames) with full spectrum
        
        time_frames = stft_result.shape[1]
        num_bins = self.resolut // 2
        
        # Create output array with shape (time_frames * num_bins, 3)
        phase_repr = np.zeros((time_frames * num_bins, 3), dtype=np.float64)
        
        # For each time frame and frequency bin:
        # v0 = spectrum[j+1], v1 = spectrum[resolut-j-1]
        # realn1 = imag(v0), realm0 = real(v1), realm1 = imag(v1)
        for t in range(time_frames):
            for j in range(num_bins):
                v0 = stft_result[j + 1, t]
                v1 = stft_result[self.resolut - j - 1, t]
                
                realn1 = np.imag(v0)
                realm0 = np.real(v1)
                realm1 = np.imag(v1)
                
                idx = t * num_bins + j
                phase_repr[idx, 0] = realn1
                phase_repr[idx, 1] = realm0
                phase_repr[idx, 2] = realm1
        
        # Subtask 5.3: Apply shrink and normalization
        # Shrink from resolut/2 bins to num_freqs bins
        shrunken = shrink(phase_repr, self.resolut, self.num_freqs)
        
        # Apply spectral normalization (log2 transform)
        normalized = spectral_normalize(shrunken)
        
        return normalized


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
    # Take absolute value to handle negative values, apply epsilon to prevent log(0), then apply log2
    # This matches the Go implementation which checks if values are < epsilon and sets them to epsilon
    normalized = np.copy(spectrogram)
    normalized = np.where(np.abs(normalized) < epsilon, epsilon, np.abs(normalized))
    return np.log2(normalized)


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


def load_wav(file_path):
    """
    Load WAV file as mono float64 array.
    
    Args:
        file_path: Path to WAV file
        
    Returns:
        1D numpy array of float64 audio samples
    """
    audio, sample_rate = sf.read(file_path, dtype='float64')
    
    # Convert stereo to mono by averaging channels if needed
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    return audio


def load_flac(file_path):
    """
    Load FLAC file as mono float64 array.
    
    Args:
        file_path: Path to FLAC file
        
    Returns:
        1D numpy array of float64 audio samples
    """
    audio, sample_rate = sf.read(file_path, dtype='float64')
    
    # Convert stereo to mono by averaging channels if needed
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    return audio


def load_wav_with_sr(file_path):
    """
    Load WAV file and return audio buffer with sample rate.
    
    Args:
        file_path: Path to WAV file
        
    Returns:
        Tuple of (audio_buffer, sample_rate) where audio_buffer is a 1D numpy array
        of float64 samples and sample_rate is an integer
    """
    audio, sample_rate = sf.read(file_path, dtype='float64')
    
    # Convert stereo to mono by averaging channels if needed
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    return audio, sample_rate


def load_flac_with_sr(file_path):
    """
    Load FLAC file and return audio buffer with sample rate.
    
    Args:
        file_path: Path to FLAC file
        
    Returns:
        Tuple of (audio_buffer, sample_rate) where audio_buffer is a 1D numpy array
        of float64 samples and sample_rate is an integer
    """
    audio, sample_rate = sf.read(file_path, dtype='float64')
    
    # Convert stereo to mono by averaging channels if needed
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    return audio, sample_rate


def save_wav(file_path, audio_buffer, sample_rate):
    """
    Save audio buffer as mono WAV file.
    
    Args:
        file_path: Path to output WAV file
        audio_buffer: 1D numpy array of audio samples
        sample_rate: Sample rate in Hz
    """
    # Clip audio values to [-1, 1] range to prevent distortion
    clipped_audio = np.clip(audio_buffer, -1.0, 1.0)
    
    # Save as 16-bit PCM mono WAV
    sf.write(file_path, clipped_audio, sample_rate, subtype='PCM_16')
