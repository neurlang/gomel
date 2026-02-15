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
            self.family = None  # Unknown family when num_freqs is explicitly set
        elif sample_rate is not None:
            # Determine num_freqs from sample rate
            if sample_rate in [8000, 16000, 24000, 32000, 48000]:
                self.num_freqs = 768 * 2
                self.family = True
            elif sample_rate in [11025, 22050, 44100]:
                self.num_freqs = 836 * 2
                self.family = False
            else:
                raise ValueError(
                    f"Unsupported sample rate: {sample_rate}. "
                    f"Supported rates are: 8000, 16000, 24000, 32000, 48000, 11025, 22050, 44100"
                )
        else:
            raise ValueError(
                f"Unset sample_rate"
                f"Please configure sample_rate to Phase"
            )
    
    def pad_shift(self, sample_rate):
        if self.family:
            if sample_rate == 48000:     
                zero_pad = 0
                zero_shift = 0
                return zero_pad, zero_shift
            if sample_rate == 32000:
                # 32000 -> 48000: 1.5x (keep 2 samples, insert 1 zero)
                zero_pad = 2
                zero_shift = 1
                return zero_pad, zero_shift
            if sample_rate == 24000:
                zero_pad = 1
                zero_shift = 1  # 24000 -> 48000: 2x (1 sample + 1 zero)
                return zero_pad, zero_shift
            if sample_rate == 16000:
                zero_pad = 1
                zero_shift = 2  # 16000 -> 48000: 3x (1 sample + 2 zeros)
                return zero_pad, zero_shift
            if sample_rate == 8000:
                zero_pad = 1
                zero_shift = 5  # 8000 -> 48000: 6x (1 sample + 5 zeros)
                return zero_pad, zero_shift
        else:
            if sample_rate == 44100:     
                zero_pad = 0
                zero_shift = 0
                return zero_pad, zero_shift
            if sample_rate == 22050:
                zero_pad = 1
                zero_shift = 1  # 22050 -> 44100: 2x (1 sample + 1 zero)
                return zero_pad, zero_shift
            if sample_rate == 11025:
                zero_pad = 1
                zero_shift = 3  # 11025 -> 44100: 4x (1 sample + 3 zeros)
                return zero_pad, zero_shift
        raise ValueError(
            f"Unsupported sample_rate"
            f"Please configure sample_rate to Phase"
        )

    def zero_pad(self, sr):
        zero_pad, _ = self.pad_shift(sr)
        return zero_pad

    def zero_shift(self, sr):
        _, zero_shift = self.pad_shift(sr)
        return zero_shift

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
        
        # gossp: stft.New(Window, Resolut) means FrameShift=Window, FrameLen=Resolut
        hop_size = self.window  # FrameShift
        frame_len = self.resolut  # FrameLen
        
        # NumFrames = int((len(input) - frameLen) / frameShift) + 1
        num_frames = int((len(padded_audio) - frame_len) / hop_size) + 1
        
        # Hann window is frameLen (resolut) long
        hann_window = np.hanning(frame_len)
        
        # Perform STFT
        stft_result = np.zeros((frame_len, num_frames), dtype=np.complex128)
        
        for i in range(num_frames):
            start = i * hop_size
            end = start + frame_len
            
            if end <= len(padded_audio):
                frame = padded_audio[start:end] * hann_window
                stft_result[:, i] = np.fft.fft(frame)
        
        # Subtask 5.2: Extract 3-channel phase representation from STFT
        time_frames = stft_result.shape[1]
        num_bins = self.resolut // 2
        
        # Create output array - Go layout is: for each time frame, append all frequency bins
        phase_repr = []
        
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
                
                phase_repr.append([realn1, realm0, realm1])
        
        phase_repr = np.array(phase_repr, dtype=np.float64)
        
        # Subtask 5.3: Apply shrink
        # Shrink from resolut/2 bins to num_freqs bins
        shrunken = shrink(phase_repr, self.resolut, self.num_freqs)
        
        return shrunken
    
    def from_phase(self, spectrogram):
        """
        Reconstruct audio from phase-preserving spectrogram.
        
        Args:
            spectrogram: 2D numpy array of shape (time_frames * num_freqs, 3)
            
        Returns:
            1D numpy array of float64 audio samples
        """
        # Subtask 6.1: Apply grow
        # Expand from num_freqs bins to resolut/2 bins
        grown = grow(spectrogram, self.resolut, self.num_freqs)
        
        # Subtask 6.2: Reconstruct complex spectrum from 3-channel representation
        num_bins = self.resolut // 2
        time_frames = len(grown) // num_bins
        
        # Create full complex spectrum array with shape (resolut, time_frames)
        spectrum = np.zeros((self.resolut, time_frames), dtype=np.complex128)
        
        # For each time frame and frequency bin:
        # Reconstruct v0 = complex(realm0, realn1) and v1 = complex(realm0, realm1)
        # Place v0 at spectrum[j+1] and v1 at spectrum[resolut-j-1]
        # Go layout is: for i in range(spectrum), for j in range(resolut/2)
        # which means index = i*(resolut/2) + j where i is time frame
        for i in range(time_frames):
            for j in range(num_bins):
                index = i * num_bins + j
                realn1 = grown[index][0]
                realm0 = grown[index][1]
                realm1 = grown[index][2]
                
                # Reconstruct complex values
                v0 = complex(realm0, realn1)
                v1 = complex(realm0, realm1)
                
                # Place in spectrum
                spectrum[j + 1, i] = v0
                spectrum[self.resolut - j - 1, i] = v1
        
        # Subtask 6.3: Implement ISTFT with proper window normalization
        hop_size = self.window  # FrameShift
        frame_len = self.resolut  # FrameLen
        
        output_len = frame_len + (time_frames - 1) * hop_size
        audio = np.zeros(output_len, dtype=np.float64)
        window_sum = np.zeros(output_len, dtype=np.float64)
        
        hann_window = np.hanning(frame_len)
        
        for i in range(time_frames):
            frame_spectrum = spectrum[:, i]
            time_domain = np.fft.ifft(frame_spectrum)
            
            start = i * hop_size
            for j in range(frame_len):
                pos = start + j
                if pos < output_len:
                    audio[pos] += np.real(time_domain[j]) * hann_window[j]
                    window_sum[pos] += hann_window[j] * hann_window[j]
        
        # Normalize by window sum (matching gossp library ISTFT)
        # Zero out edges where window overlap is insufficient to avoid spikes
        stable_threshold = window_sum.max() * 0.5
        for n in range(output_len):
            if window_sum[n] > stable_threshold:
                audio[n] /= window_sum[n]
            elif window_sum[n] > 1e-21:
                # Fade: scale down proportionally in transition zones
                audio[n] = audio[n] / window_sum[n] * (window_sum[n] / stable_threshold)
        
        # Subtask 6.4: Apply volume boost if configured
        if self.volume_boost > 0:
            audio = audio * self.volume_boost
        
        # Return as numpy float64 array
        return audio.astype(np.float64)
    
    def to_phase_wav(self, input_file, output_file):
        """
        Convert WAV file to PNG spectrogram.
        
        Args:
            input_file: Path to input WAV file
            output_file: Path to output PNG file
        """
        # Load WAV file using load_wav_with_sr()
        audio, sample_rate = load_wav_with_sr(input_file)
        
        # Apply zero stuffing upsampling if configured
        zero_pad = self.zero_pad(sample_rate)
        zero_shift = self.zero_shift(sample_rate)
        if zero_pad > 0:
            original_len = len(audio)
            audio = zero_stuff_upsample(audio, zero_pad, zero_shift)
            # Update sample rate based on actual upsampling ratio
            sample_rate = int(sample_rate * len(audio) / original_len)
        
        # Store original length for padding detection
        original_length = len(audio)
        
        # Call to_phase() to generate spectrogram
        spectrogram = self.to_phase(audio)
        
        # Calculate samples_in_mel ratio - Go: float64(len(buf)*m.NumFreqs)/float64(len(ospectrum))
        samples_in_mel = float(original_length * self.num_freqs) / float(len(spectrogram))
        
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
        
        # Apply zero stuffing upsampling if configured
        zero_pad = self.zero_pad(sample_rate)
        zero_shift = self.zero_shift(sample_rate)
        if zero_pad > 0:
            original_len = len(audio)
            audio = zero_stuff_upsample(audio, zero_pad, zero_shift)
            # Update sample rate based on actual upsampling ratio
            sample_rate = int(sample_rate * len(audio) / original_len)
        
        # Store original length for padding detection
        original_length = len(audio)
        
        # Call to_phase() to generate spectrogram
        spectrogram = self.to_phase(audio)
        
        # Calculate samples_in_mel ratio - Go: float64(len(buf)*m.NumFreqs)/float64(len(ospectrum))
        samples_in_mel = float(original_length * self.num_freqs) / float(len(spectrogram))
        
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
        spectrogram, samples, embedded_sample_rate = load_image(input_file, self.y_reverse)
        
        # Call from_phase() to reconstruct audio
        audio = self.from_phase(spectrogram)
        
        # Round embedded sample rate to nearest standard rate
        standard_rates = [8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000]
        sample_rate = min(standard_rates, key=lambda x: abs(x - embedded_sample_rate))
        
        # samples is the original audio length
        original_length = int(samples)
        
        # Trim to original length if reconstructed audio is longer
        if len(audio) > original_length > 0:
            audio = audio[:original_length]
        
        # Call save_wav() to write output file
        save_wav(output_file, audio, sample_rate)


def pad(audio_buffer, window):
    """
    Apply padding to audio buffer for proper STFT processing.
    Matches Go implementation exactly.
    
    Args:
        audio_buffer: 1D numpy array of audio samples
        window: STFT window size (filter)
        
    Returns:
        Padded audio buffer as numpy array
    """
    current_len = len(audio_buffer)
    min_target_size = 15 * window
    pad_len = 0
    
    if current_len >= min_target_size:
        remainder = (current_len - min_target_size) % window
        if remainder != 0:
            pad_len = window - remainder - 1
    else:
        pad_len = min_target_size - current_len - 1
    
    if pad_len > 0:
        return np.pad(audio_buffer, (0, pad_len), mode='constant', constant_values=0)
    return audio_buffer


def is_padded(original_length, padded_length, window):
    """
    Detect if audio was padded based on length comparison.
    Matches Go implementation exactly.
    
    Args:
        original_length: Original audio buffer length
        padded_length: Length after padding
        window: STFT window size (filter)
        
    Returns:
        True if audio was padded, False otherwise
    """
    min_target_size = 15 * window
    
    if original_length >= min_target_size:
        remainder = (original_length - min_target_size) % window
        if remainder != 0:
            pad_len = window - remainder - 1
            return padded_length == original_length + pad_len
        else:
            return padded_length == original_length
    else:
        pad_len = min_target_size - original_length - 1
        return padded_length == original_length + pad_len


def spectral_normalize(spectrogram):
    """
    Apply spectral normalization using log2 transformation.
    
    Args:
        spectrogram: 2D numpy array of shape (time_frames * num_freqs, 3)
        
    Returns:
        Normalized spectrogram with log2 transformation applied
    """
    epsilon = 1e-10
    # Match Go: if value < epsilon (including negatives), clamp to epsilon, then log2
    normalized = np.copy(spectrogram)
    normalized = np.where(normalized < epsilon, epsilon, normalized)
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
    # Go implementation: keep only entries where (i % imels) < omels
    # This means: for each time frame, keep only the first num_freqs frequency bins
    original_bins = resolut // 2
    
    out = []
    for i in range(len(spectrogram)):
        j = i % original_bins
        if j < num_freqs:
            out.append(spectrogram[i])
    
    return np.array(out, dtype=spectrogram.dtype)


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
    # Go implementation: append each entry, and when we reach the end of a frame (j+1 == imels),
    # replicate the last entry to fill the rest of the frame
    target_bins = resolut // 2
    
    out = []
    for i in range(len(spectrogram)):
        j = i % num_freqs
        out.append(spectrogram[i])
        # If we just added the last frequency bin of this time frame
        if j + 1 == num_freqs:
            # Replicate it to fill the remaining bins
            for k in range(num_freqs, target_bins):
                out.append(spectrogram[i])
    
    return np.array(out, dtype=spectrogram.dtype)


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


def zero_stuff_upsample(audio, zero_pad, zero_shift):
    """
    Upsample audio by inserting zeros between samples.
    
    Args:
        audio: 1D numpy array of audio samples
        zero_pad: Number of samples to keep before inserting zeros
        zero_shift: Number of zeros to insert after each zero_pad samples
        
    Returns:
        Upsampled audio buffer with zeros inserted
        
    Examples:
        - 22050 -> 44100: zero_pad=1, zero_shift=1 (keep 1 sample, pad 1 zero) = 2x
        - 11025 -> 44100: zero_pad=1, zero_shift=3 (keep 1 sample, pad 3 zeros) = 4x
        - 32000 -> 48000: zero_pad=2, zero_shift=1 (keep 2 samples, pad 1 zero) = 1.5x
    """
    if zero_pad == 0:
        return audio
    
    # Calculate output length
    # For every zero_pad samples, we add zero_shift zeros
    num_groups = (len(audio) + zero_pad - 1) // zero_pad
    output_len = len(audio) + num_groups * zero_shift
    output = np.zeros(output_len, dtype=audio.dtype)
    
    # Insert original samples with zeros in between
    out_idx = 0
    for i in range(len(audio)):
        output[out_idx] = audio[i]
        out_idx += 1
        # After every zero_pad samples, insert zero_shift zeros
        if (i + 1) % zero_pad == 0:
            out_idx += zero_shift
    
    return output


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
    stride = len(spectrogram) // num_freqs
    
    # Calculate max/min values for each of 3 channels
    max_values = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    min_values = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    
    for x in range(stride):
        for l in range(3):
            for y in range(num_freqs):
                w = spectrogram[y + x * num_freqs][l]
                if w > max_values[l]:
                    max_values[l] = w
                if w < min_values[l]:
                    min_values[l] = w
    
    # Pack metadata: [max0, max1, max2, min0, min1, min2, samples_in_mel, sample_rate]
    metadata = [
        max_values[0], max_values[1], max_values[2],
        min_values[0], min_values[1], min_values[2],
        samples_in_mel, sample_rate
    ]
    
    floats = []
    for value in metadata:
        floats.extend(pack_float16_to_bytes(value))
    
    # Create image array with shape (num_freqs, stride, 3)
    image_data = np.zeros((num_freqs, stride, 3), dtype=np.uint8)
    
    # Normalize each channel to 0-255 range
    for x in range(stride):
        for y in range(num_freqs):
            idx = y + x * num_freqs  # Go layout: buf[y+x*mels]
            
            for ch in range(3):
                channel_range = max_values[ch] - min_values[ch]
                if channel_range > 0:
                    val = (spectrogram[idx][ch] - min_values[ch]) / channel_range
                    image_data[y, x, ch] = int(255 * val)
                else:
                    image_data[y, x, ch] = 128
            
            # Embed metadata in first column (x=0) blue channel
            if x == 0 and y < len(floats):
                image_data[y, x, 2] = floats[y]
    
    # Create PIL Image
    img = Image.fromarray(image_data, mode='RGB')
    
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
        Tuple of (spectrogram, samples, sample_rate) where:
        - spectrogram: 2D numpy array of shape (time_frames * num_freqs, 3)
        - samples: Original audio length in samples
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
    
    # Get dimensions
    width, height = img.size
    stride = width  # time frames
    num_freqs = height
    
    # Get pixel access
    pixels = img.load()
    
    # Extract metadata from first column (x=0) blue channel
    floats = []
    for y in range(min(32, num_freqs)):
        r, g, b = pixels[0, y]
        floats.append(b)
    
    # Pad with zeros if needed
    while len(floats) < 16:
        floats.append(0)
    
    # Unpack 8 float16 values for metadata
    metadata = []
    for i in range(8):
        byte_pair = bytes([floats[i * 2], floats[i * 2 + 1]])
        value = unpack_bytes_to_float64(byte_pair)
        metadata.append(value)
    
    max_values = np.array([metadata[0], metadata[1], metadata[2]])
    min_values = np.array([metadata[3], metadata[4], metadata[5]])
    samples_in_mel = metadata[6]
    sample_rate = int(metadata[7])
    
    # Convert image to numpy array
    img_array = np.array(img)  # Shape: (height, width, 3) = (num_freqs, stride, 3)
    
    # Create output buffer matching Go layout: buf[y+x*mels]
    buf = []
    for x in range(stride):
        for y in range(num_freqs):
            val0 = float(img_array[y, x, 0]) / 255.0
            val1 = float(img_array[y, x, 1]) / 255.0
            val2 = float(img_array[y, x, 2]) / 255.0
            buf.append([val0, val1, val2])
    
    buf = np.array(buf, dtype=np.float64)
    
    # Denormalize from 0-1 to original range using metadata
    for i in range(len(buf)):
        buf[i][0] = (buf[i][0] * (max_values[0] - min_values[0]) + min_values[0])
        buf[i][1] = (buf[i][1] * (max_values[1] - min_values[1]) + min_values[1])
        buf[i][2] = (buf[i][2] * (max_values[2] - min_values[2]) + min_values[2])
    
    # Calculate samples from samples_in_mel (matching Go: samples = samples_in_mel * stride)
    samples = samples_in_mel * stride
    
    return buf, samples, sample_rate
