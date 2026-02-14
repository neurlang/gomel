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
    
    def from_phase(self, spectrogram):
        """
        Reconstruct audio from phase-preserving spectrogram.
        
        Args:
            spectrogram: 2D numpy array of shape (time_frames * num_freqs, 3)
            
        Returns:
            1D numpy array of float64 audio samples
        """
        # Subtask 6.1: Apply denormalization and grow
        # Apply spectral denormalization (exp2 transform)
        denormalized = spectral_denormalize(spectrogram)
        
        # Expand from num_freqs bins to resolut/2 bins
        grown = grow(denormalized, self.resolut, self.num_freqs)
        
        # Subtask 6.2: Reconstruct complex spectrum from 3-channel representation
        num_bins = self.resolut // 2
        time_frames = len(grown) // num_bins
        
        # Create full complex spectrum array with shape (resolut, time_frames)
        spectrum = np.zeros((self.resolut, time_frames), dtype=np.complex128)
        
        # For each time frame and frequency bin:
        # Reconstruct v0 = complex(realm0, realn1) and v1 = complex(realm0, realm1)
        # Place v0 at spectrum[j+1] and v1 at spectrum[resolut-j-1]
        for t in range(time_frames):
            for j in range(num_bins):
                idx = t * num_bins + j
                realn1 = grown[idx, 0]
                realm0 = grown[idx, 1]
                realm1 = grown[idx, 2]
                
                # Reconstruct complex values
                v0 = complex(realm0, realn1)
                v1 = complex(realm0, realm1)
                
                # Place in spectrum
                spectrum[j + 1, t] = v0
                spectrum[self.resolut - j - 1, t] = v1
        
        # Subtask 6.3: Implement ISTFT computation
        # Use scipy.signal.istft with matching parameters
        hop_size = self.window // 4
        noverlap = self.window - hop_size
        
        times, audio = signal.istft(
            spectrum,
            window='hann',
            nperseg=self.window,
            noverlap=noverlap,
            nfft=self.resolut,
            input_onesided=False
        )
        
        # Subtask 6.4: Apply volume boost if configured
        if self.volume_boost > 0:
            audio = audio * self.volume_boost
        
        # Return as numpy float64 array (take real part to handle numerical precision)
        return np.real(audio).astype(np.float64)
    
    def to_phase_wav(self, input_file, output_file):
        """
        Convert WAV file to PNG spectrogram.
        
        Args:
            input_file: Path to input WAV file
            output_file: Path to output PNG file
        """
        # Load WAV file using load_wav_with_sr()
        audio, sample_rate = load_wav_with_sr(input_file)
        
        # Store original length for padding detection
        original_length = len(audio)
        
        # Call to_phase() to generate spectrogram
        spectrogram = self.to_phase(audio)
        
        # Calculate samples_in_mel ratio
        # This is the ratio of original audio samples to spectrogram length
        samples_in_mel = original_length / len(spectrogram)
        
        # Call save_image() with spectrogram and metadata
        save_image(output_file, spectrogram, self.num_freqs, samples_in_mel, 
                   sample_rate, self.y_reverse)
    
    def to_phase_flac(self, input_file, output_file):
        """
        Convert FLAC file to PNG spectrogram.
        
        Args:
            input_file: Path to input FLAC file
            output_file: Path to output PNG file
        """
        # Load FLAC file using load_flac_with_sr()
        audio, sample_rate = load_flac_with_sr(input_file)
        
        # Store original length for padding detection
        original_length = len(audio)
        
        # Call to_phase() to generate spectrogram
        spectrogram = self.to_phase(audio)
        
        # Calculate samples_in_mel ratio
        # This is the ratio of original audio samples to spectrogram length
        samples_in_mel = original_length / len(spectrogram)
        
        # Call save_image() with spectrogram and metadata
        save_image(output_file, spectrogram, self.num_freqs, samples_in_mel, 
                   sample_rate, self.y_reverse)
    
    def to_wav_png(self, input_file, output_file):
        """
        Convert PNG spectrogram to WAV file.
        
        Args:
            input_file: Path to input PNG file
            output_file: Path to output WAV file
        """
        # Load PNG using load_image() to get spectrogram and metadata
        spectrogram, samples_in_mel, embedded_sample_rate = load_image(input_file, self.y_reverse)
        
        # Call from_phase() to reconstruct audio
        audio = self.from_phase(spectrogram)
        
        # Use embedded sample rate if self.sample_rate is not set
        if self.sample_rate is not None:
            sample_rate = self.sample_rate
        else:
            sample_rate = embedded_sample_rate
        
        # Calculate original length from samples_in_mel ratio
        original_length = int(samples_in_mel * len(spectrogram))
        
        # Trim padding if original length is known using is_padded()
        if is_padded(original_length, len(audio), self.window):
            # Trim to original length
            audio = audio[:original_length]
        
        # Call save_wav() to write output file
        save_wav(output_file, audio, sample_rate)


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


def pack_float16_to_bytes(value):
    """
    Convert float64 to float16 and pack as 2 bytes in little-endian format.
    
    Args:
        value: Float64 value to pack
        
    Returns:
        2 bytes representing the float16 value in little-endian format
    """
    # Convert float64 to float16 using numpy
    float16_value = np.float16(value)
    
    # Pack as 2 bytes in little-endian format
    # 'e' format code is for float16 (half precision)
    return struct.pack('<e', float16_value)


def unpack_bytes_to_float64(byte_data):
    """
    Unpack 2 bytes as little-endian float16 and convert to float64.
    
    Args:
        byte_data: 2 bytes representing a float16 value in little-endian format
        
    Returns:
        Float64 value
    """
    # Unpack 2 bytes as little-endian float16
    float16_value = struct.unpack('<e', byte_data)[0]
    
    # Convert to float64
    return np.float64(float16_value)


def save_image(file_path, spectrogram, num_freqs, samples_in_mel, sample_rate, y_reverse=True):
    """
    Save phase-preserving spectrogram as PNG image with embedded metadata.
    
    Args:
        file_path: Path to output PNG file
        spectrogram: 2D numpy array of shape (time_frames * num_freqs, 3)
        num_freqs: Number of frequency bins
        samples_in_mel: Ratio of samples to mel (for reconstruction)
        sample_rate: Audio sample rate
        y_reverse: Flip Y-axis if True (default: True)
    """
    # Calculate stride (time frames) from spectrogram length and num_freqs
    time_frames = len(spectrogram) // num_freqs
    
    # Calculate max/min values for each of 3 channels
    max_values = np.max(spectrogram, axis=0)  # Shape: (3,)
    min_values = np.min(spectrogram, axis=0)  # Shape: (3,)
    
    # Normalize each channel to 0-255 range
    normalized = np.zeros_like(spectrogram, dtype=np.uint8)
    for ch in range(3):
        channel_range = max_values[ch] - min_values[ch]
        if channel_range > 0:
            normalized[:, ch] = ((spectrogram[:, ch] - min_values[ch]) / channel_range * 255).astype(np.uint8)
        else:
            # If all values are the same, set to 128
            normalized[:, ch] = 128
    
    # Reshape to (time_frames, num_freqs, 3) for image creation
    reshaped = normalized.reshape(time_frames, num_freqs, 3)
    
    # Create RGB image: R=channel0, G=channel1, B=channel2
    # PIL expects (height, width, channels), so we transpose to (num_freqs, time_frames, 3)
    image_data = np.transpose(reshaped, (1, 0, 2))
    
    # Create PIL Image
    img = Image.fromarray(image_data, mode='RGB')
    
    # Pack metadata: [max0, max1, max2, min0, min1, min2, samples_in_mel, sample_rate]
    metadata = [
        max_values[0], max_values[1], max_values[2],
        min_values[0], min_values[1], min_values[2],
        samples_in_mel, sample_rate
    ]
    
    # Convert image to mutable pixel access
    pixels = img.load()
    
    # Embed metadata in first column (x=0) blue channel pixels 0-15
    # Each float16 takes 2 bytes, stored in 2 consecutive pixels' blue channels
    for i, value in enumerate(metadata):
        packed_bytes = pack_float16_to_bytes(value)
        # Each float16 takes 2 bytes, so we need 2 pixels per value
        pixel_idx_1 = i * 2
        pixel_idx_2 = i * 2 + 1
        
        if pixel_idx_1 < num_freqs and pixel_idx_2 < num_freqs:
            # Get current pixel values (R and G channels contain spectrogram data)
            r1, g1, b1 = pixels[0, pixel_idx_1]
            r2, g2, b2 = pixels[0, pixel_idx_2]
            
            # Replace blue channel with metadata bytes, keep R and G unchanged
            pixels[0, pixel_idx_1] = (r1, g1, packed_bytes[0])
            pixels[0, pixel_idx_2] = (r2, g2, packed_bytes[1])
    
    # Apply Y-axis flip if y_reverse=True
    if y_reverse:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    
    # Save as PNG
    img.save(file_path, format='PNG')


def load_image(file_path, y_reverse=True):
    """
    Load phase-preserving spectrogram from PNG image with embedded metadata.
    
    Args:
        file_path: Path to PNG file
        y_reverse: Flip Y-axis if True (default: True)
        
    Returns:
        Tuple of (spectrogram, samples_in_mel, sample_rate) where:
        - spectrogram: 2D numpy array of shape (time_frames * num_freqs, 3)
        - samples_in_mel: Ratio of samples to mel
        - sample_rate: Audio sample rate
    """
    # Load PNG with PIL
    img = Image.open(file_path)
    
    # Apply Y-axis flip if y_reverse=True (undo the flip from save)
    if y_reverse:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    
    # Convert to RGB if not already
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Get pixel access
    pixels = img.load()
    width, height = img.size
    num_freqs = height
    time_frames = width
    
    # Extract metadata from first column (x=0) blue channel pixels 0-15
    # Metadata: [max0, max1, max2, min0, min1, min2, samples_in_mel, sample_rate]
    metadata_bytes = []
    for i in range(min(16, num_freqs)):  # 8 float16 values * 2 bytes each = 16 bytes
        r, g, b = pixels[0, i]
        metadata_bytes.append(b)
    
    # Pad with zeros if we don't have enough pixels
    while len(metadata_bytes) < 16:
        metadata_bytes.append(0)
    
    # Unpack 8 float16 values for metadata
    metadata = []
    for i in range(8):
        byte_pair = bytes([metadata_bytes[i * 2], metadata_bytes[i * 2 + 1]])
        value = unpack_bytes_to_float64(byte_pair)
        metadata.append(value)
    
    max_values = np.array([metadata[0], metadata[1], metadata[2]])
    min_values = np.array([metadata[3], metadata[4], metadata[5]])
    samples_in_mel = metadata[6]
    sample_rate = int(metadata[7])
    
    # Convert image to numpy array
    img_array = np.array(img)  # Shape: (height, width, 3) = (num_freqs, time_frames, 3)
    
    # Transpose to (time_frames, num_freqs, 3)
    transposed = np.transpose(img_array, (1, 0, 2))
    
    # Reshape to (time_frames * num_freqs, 3)
    flattened = transposed.reshape(time_frames * num_freqs, 3)
    
    # Denormalize from 0-255 to original range using metadata
    spectrogram = np.zeros_like(flattened, dtype=np.float64)
    for ch in range(3):
        channel_range = max_values[ch] - min_values[ch]
        spectrogram[:, ch] = (flattened[:, ch].astype(np.float64) / 255.0) * channel_range + min_values[ch]
    
    return spectrogram, samples_in_mel, sample_rate
