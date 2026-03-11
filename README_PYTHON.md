# phase-spectrogram

Phase-preserving spectrogram encoder/decoder for high-quality audio reconstruction.

This Python package implements a phase-preserving spectrogram encoder/decoder that converts audio waveforms to spectrograms and back to audio without loss of phase information, enabling high-quality audio reconstruction.

## Installation

```bash
pip install phase-spectrogram
```

## Quick Start

```python
from phase import Phase

# Initialize with sample rate
phase = Phase(sample_rate=44100)

# Convert audio file to spectrogram image
phase.to_phase_wav('input.wav', 'output.png')

# Convert spectrogram image back to audio
phase.to_wav_png('output.png', 'reconstructed.wav')
```

## Features

- **Phase-Preserving**: Retains both magnitude and phase information for lossless reconstruction
- **High-Quality Audio**: Near-lossless audio reconstruction without iterative algorithms
- **Multiple Sample Rates**: Support for 8kHz, 11.025kHz, 16kHz, 22.05kHz, 24kHz, 32kHz, 44.1kHz, and 48kHz
- **Flexible Formats**: WAV and FLAC input support
- **PNG Export**: Save spectrograms as images for visualization or ML applications
- **HDR Support**: Optional 16-bit per channel PNG for higher dynamic range

## Usage Examples

### Basic Audio Processing

```python
import numpy as np
from phase import Phase

# Create Phase encoder/decoder
phase = Phase(sample_rate=44100)

# Generate test audio (1 second of 440Hz sine wave)
t = np.linspace(0, 1.0, 44100)
audio = np.sin(2 * np.pi * 440 * t)

# Convert to spectrogram
spectrogram = phase.to_phase(audio)

# Reconstruct audio
reconstructed = phase.from_phase(spectrogram)
```

### File Conversion

```python
from phase import Phase

phase = Phase(sample_rate=44100)

# WAV to PNG
phase.to_phase_wav('input.wav', 'spectrogram.png')

# FLAC to PNG
phase.to_phase_flac('input.flac', 'spectrogram.png')

# PNG back to WAV
phase.to_wav_png('spectrogram.png', 'output.wav')
```

### Advanced Configuration

```python
from phase import Phase

# High Dynamic Range (16-bit per channel)
phase_hdr = Phase(
    sample_rate=48000,
    HDR=True,
    volume_boost=2.0,
    y_reverse=False
)

# Custom window and FFT resolution
phase_custom = Phase(
    sample_rate=44100,
    window=2560,
    resolut=8192
)

# With inverse hyperbolic sine compression
phase_ihs = Phase(
    sample_rate=44100,
    IHS=True
)
```

## API Reference

### Phase Class

```python
Phase(sample_rate=None, num_freqs=None, window=1280, 
      resolut=4096, y_reverse=True, volume_boost=0.0, 
      HDR=False, IHS=False)
```

**Parameters:**
- `sample_rate` (int): Audio sample rate (8000, 11025, 16000, 22050, 24000, 32000, 44100, or 48000)
- `num_freqs` (int): Number of frequency bins (auto-set based on sample_rate if not provided)
- `window` (int): STFT window size (default: 1280)
- `resolut` (int): FFT resolution (default: 4096)
- `y_reverse` (bool): Flip Y-axis in PNG images (default: True)
- `volume_boost` (float): Volume multiplier for reconstruction (default: 0.0 = no boost)
- `HDR` (bool): Use 16 bits per channel PNG (default: False = 8 bits per channel)
- `IHS` (bool): Enable inverse hyperbolic sine compression (default: False)

### Methods

#### `to_phase(audio_buffer)`
Convert audio buffer to phase-preserving spectrogram.

**Parameters:**
- `audio_buffer` (numpy.ndarray): 1D array of float64 audio samples

**Returns:**
- `numpy.ndarray`: 2D array of shape (time_frames * num_freqs, 2)

#### `from_phase(spectrogram)`
Reconstruct audio from phase-preserving spectrogram.

**Parameters:**
- `spectrogram` (numpy.ndarray): 2D array of shape (time_frames * num_freqs, 2)

**Returns:**
- `numpy.ndarray`: 1D array of float64 audio samples

#### `to_phase_wav(input_file, output_file)`
Convert WAV file to PNG spectrogram.

**Parameters:**
- `input_file` (str): Path to input WAV file
- `output_file` (str): Path to output PNG file

#### `to_phase_flac(input_file, output_file)`
Convert FLAC file to PNG spectrogram.

**Parameters:**
- `input_file` (str): Path to input FLAC file
- `output_file` (str): Path to output PNG file

#### `to_wav_png(input_file, output_file)`
Convert PNG spectrogram to WAV file.

**Parameters:**
- `input_file` (str): Path to input PNG file
- `output_file` (str): Path to output WAV file

**Returns:**
- `int`: Detected sample rate from the spectrogram

## Supported Sample Rates

| Sample Rate | Frequency Bins | Family |
|-------------|----------------|--------|
| 8000 Hz     | 768            | 48k    |
| 16000 Hz    | 768            | 48k    |
| 24000 Hz    | 768            | 48k    |
| 32000 Hz    | 768            | 48k    |
| 48000 Hz    | 768            | 48k    |
| 11025 Hz    | 836            | 44.1k  |
| 22050 Hz    | 836            | 44.1k  |
| 44100 Hz    | 836            | 44.1k  |

Note: HDR mode doubles the frequency bin count.

## Requirements

- Python >= 3.7
- numpy >= 1.20.0
- scipy >= 1.7.0
- soundfile >= 0.10.0
- Pillow >= 8.0.0
- pypng >= 0.20220715.0

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Related Projects

This is a Python implementation based on the Go package [gomel](https://github.com/neurlang/gomel).
