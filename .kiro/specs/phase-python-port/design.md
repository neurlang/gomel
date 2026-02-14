# Design Document: Phase-Preserving Spectrogram Python Port

## Overview

This design document describes the Python implementation of the phase-preserving spectrogram encoder/decoder. The system will use NumPy for numerical operations, SciPy for signal processing (STFT/ISTFT), librosa or soundfile for audio I/O, and Pillow (PIL) for PNG image handling. The implementation will maintain API compatibility with the Go version while following Python conventions.

## Architecture

The system consists of a single main module `phase.py` containing:

1. **Phase class**: Main interface for configuration and high-level operations
2. **Core transformation functions**: STFT-based encoding and decoding
3. **Audio I/O functions**: Loading and saving WAV/FLAC files
4. **Image I/O functions**: Saving and loading PNG spectrograms with metadata
5. **Utility functions**: Padding, normalization, spectrum manipulation
6. **Raw functions**: Saving and loading raw float spectrograms

### Dependencies

- `numpy`: Numerical operations and array handling
- `scipy.signal`: STFT and ISTFT implementations
- `soundfile` or `librosa`: Audio file I/O (WAV/FLAC support)
- `Pillow (PIL)`: PNG image encoding/decoding
- `struct`: Binary packing for float16 metadata encoding

## Components and Interfaces

### Phase Class

```python
class Phase:
    def __init__(self, sample_rate=None, window=1280, 
                 resolut=4096, y_reverse=True, volume_boost=0.0):
        """
        Initialize Phase encoder/decoder.
        
        Args:
            sample_rate: Audio sample rate (determines num_freqs if not provided)
            num_freqs: Number of frequency bins (auto-set based on sample_rate)
            window: STFT window size (default: 1280)
            resolut: FFT resolution (default: 4096)
            y_reverse: Flip Y-axis in PNG images (default: True)
            volume_boost: Volume multiplier for reconstruction (default: 0.0 = no boost)
        """
```

### Core Methods

```python
def to_phase(self, audio_buffer: np.ndarray) -> np.ndarray:
    """
    Convert audio buffer to phase-preserving spectrogram.
    
    Args:
        audio_buffer: 1D numpy array of float64 audio samples
        
    Returns:
        2D numpy array of shape (time_frames * num_freqs, 3) containing
        [realn1, realm0, realm1] for each time-frequency bin
    """

def from_phase(self, spectrogram: np.ndarray) -> np.ndarray:
    """
    Reconstruct audio from phase-preserving spectrogram.
    
    Args:
        spectrogram: 2D numpy array of shape (time_frames * num_freqs, 3)
        
    Returns:
        1D numpy array of float64 audio samples
    """
```

### Audio I/O Methods

```python
def load_wav(file_path: str) -> np.ndarray:
    """Load mono WAV file as float64 array."""

def load_flac(file_path: str) -> np.ndarray:
    """Load mono FLAC file as float64 array."""

def load_wav_with_sr(file_path: str) -> tuple[np.ndarray, int]:
    """Load WAV file and return (audio, sample_rate)."""

def load_flac_with_sr(file_path: str) -> tuple[np.ndarray, int]:
    """Load FLAC file and return (audio, sample_rate)."""

def save_wav(file_path: str, audio_buffer: np.ndarray, sample_rate: int):
    """Save audio buffer as mono WAV file."""
```

### Image I/O Methods

```python
def to_phase_wav(self, input_file: str, output_file: str):
    """Convert WAV file to PNG spectrogram."""

def to_phase_flac(self, input_file: str, output_file: str):
    """Convert FLAC file to PNG spectrogram."""

def to_wav_png(self, input_file: str, output_file: str):
    """Convert PNG spectrogram to WAV file."""
```

## Data Models

### Spectrogram Representation

The phase-preserving spectrogram is stored as a 2D numpy array with shape `(time_frames * num_freqs, 3)`:
- Channel 0 (`realn1`): Imaginary component of positive frequency
- Channel 1 (`realm0`): Real component of positive frequency  
- Channel 2 (`realm1`): Imaginary component of negative frequency (conjugate)

This 3-channel representation preserves full phase information for perfect reconstruction.

### PNG Metadata Encoding

Metadata is embedded in the first column (x=0) of the PNG image in the blue channel (first 16 pixels):
- Pixels 0-1: max value channel 0 (float16)
- Pixels 2-3: max value channel 1 (float16)
- Pixels 4-5: max value channel 2 (float16)
- Pixels 6-7: min value channel 0 (float16)
- Pixels 8-9: min value channel 1 (float16)
- Pixels 10-11: min value channel 2 (float16)
- Pixels 12-13: samples_in_mel ratio (float16)
- Pixels 14-15: sample rate (float16)

Each float16 value is encoded as 2 bytes in little-endian format.

## Data Flow

### Encoding (Audio → PNG)

1. Load audio file (WAV/FLAC) as float64 mono samples
2. Apply padding to ensure proper STFT framing
3. Compute STFT with configured window and resolution
4. Extract 3-channel phase representation from complex spectrum
5. Shrink from Resolut/2 bins to NumFreqs bins
6. Apply spectral normalization (log2 transform)
7. Normalize to 0-255 range per channel
8. Embed metadata in first column blue channel
9. Save as PNG image (with optional Y-axis flip)

### Decoding (PNG → Audio)

1. Load PNG image
2. Extract metadata from first column blue channel
3. Decode RGB pixels to 3-channel spectrogram
4. Denormalize from 0-255 to original range using metadata
5. Apply spectral denormalization (exp2 transform)
6. Grow from NumFreqs bins to Resolut/2 bins
7. Reconstruct complex spectrum from 3-channel representation
8. Compute ISTFT to generate audio samples
9. Apply volume boost if configured
10. Trim padding if original length is known
11. Save as WAV file



## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Audio-to-spectrogram-to-audio round trip preserves signal

*For any* audio buffer, converting to phase spectrogram and back to audio should produce a signal that is perceptually similar to the original (allowing for numerical precision and padding effects).

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 3.1, 3.2, 3.3, 3.4**

### Property 2: PNG round trip preserves spectrogram data

*For any* phase spectrogram, saving to PNG and loading back should preserve the spectrogram values within float16 precision limits.

**Validates: Requirements 6.1, 6.2, 6.3, 6.5, 7.1, 7.2, 7.3, 7.5**

### Property 3: Configuration parameters are respected

*For any* valid configuration values (Window, Resolut, NumFreqs, YReverse, VolumeBoost), setting these parameters should result in them being used during processing.

**Validates: Requirements 1.4**

### Property 4: Spectrogram output has correct dimensions

*For any* audio buffer, the output spectrogram from to_phase should have shape (time_frames * num_freqs, 3) where time_frames depends on the audio length and window parameters.

**Validates: Requirements 2.4, 2.6**

### Property 5: Volume boost scales output

*For any* audio buffer and volume boost value > 0, the reconstructed audio with volume boost should have amplitudes scaled by the boost factor compared to reconstruction without boost.

**Validates: Requirements 3.5**

### Property 6: WAV save/load round trip preserves audio

*For any* audio buffer with values in [-1, 1], saving to WAV and loading back should preserve the audio within 16-bit PCM quantization limits.

**Validates: Requirements 5.1, 5.2**

### Property 7: Audio clipping prevents out-of-range values

*For any* audio buffer with values outside [-1, 1], saving to WAV should clip values to the valid range.

**Validates: Requirements 5.3**

### Property 8: Y-axis reversal is consistent

*For any* spectrogram, saving with y_reverse=True and loading with y_reverse=True should preserve the spectrogram, while mismatched settings should produce a vertically flipped result.

**Validates: Requirements 6.4, 7.4**

### Property 9: Metadata embedding preserves reconstruction parameters

*For any* spectrogram with associated metadata (max/min values, sample rate, samples ratio), the metadata embedded in PNG should be accurately recovered when loading.

**Validates: Requirements 6.2, 6.3, 7.2, 7.3**

### Property 10: Padding is correctly applied and removed

*For any* audio buffer, the padding applied before STFT should be correctly identified and removed during reconstruction when original length metadata is available.

**Validates: Requirements 2.1, 8.5**

## Error Handling

### File I/O Errors

- **Missing files**: Raise `FileNotFoundError` with descriptive message
- **Invalid audio format**: Raise `ValueError` with format details
- **Corrupted files**: Raise appropriate exception from underlying library (soundfile, PIL)
- **Permission errors**: Propagate OS-level permission exceptions

### Invalid Parameters

- **Invalid sample rate**: Raise `ValueError` if sample rate is not positive
- **Invalid dimensions**: Raise `ValueError` if audio buffer or spectrogram has wrong shape
- **Invalid configuration**: Raise `ValueError` for non-positive Window or Resolut values

### Numerical Issues

- **Division by zero**: Apply epsilon (1e-10) before log operations to prevent log(0)
- **NaN/Inf values**: Check for and raise `ValueError` if detected in critical paths
- **Overflow**: Clip audio values to [-1, 1] range before WAV export

## Testing Strategy

### Unit Testing

The implementation will include unit tests for:

- **Initialization**: Test default values and custom configuration
- **Padding logic**: Test pad() function with various buffer sizes
- **Normalization**: Test spectral_normalize() and spectral_denormalize() are inverses
- **Shrink/grow**: Test spectrum dimension manipulation
- **Metadata encoding**: Test float16 packing and unpacking
- **File I/O**: Test loading and saving with known test files
- **Error cases**: Test exception handling for invalid inputs

### Property-Based Testing

The implementation will use Hypothesis (Python property-based testing library) to verify:

- **Round-trip properties**: Audio → spectrogram → audio preservation
- **PNG round-trip**: Spectrogram → PNG → spectrogram preservation  
- **Configuration respect**: Random valid configs produce expected behavior
- **Dimension correctness**: Output shapes match expected dimensions
- **Volume boost scaling**: Linear scaling relationship
- **Clipping behavior**: Out-of-range values are properly clipped
- **Metadata preservation**: Embedded metadata is accurately recovered

Each property-based test will run a minimum of 100 iterations with randomly generated inputs to ensure robustness across the input space.

### Integration Testing

- **End-to-end workflows**: Test complete pipelines (WAV → PNG → WAV)
- **Cross-format compatibility**: Test with various sample rates and audio lengths
- **Comparison with Go implementation**: Verify outputs match the original Go version

### Test Data

- **Synthetic signals**: Sine waves, white noise, chirps for controlled testing
- **Real audio samples**: Short clips at various sample rates (8kHz, 16kHz, 44.1kHz, 48kHz)
- **Edge cases**: Empty audio, single sample, very long audio

## Implementation Notes

### NumPy Conventions

- Use `np.float64` for audio buffers (matches Go's float64)
- Use `np.complex128` for STFT output (matches Go's complex128)
- Follow NumPy broadcasting rules for efficient operations

### STFT Configuration

- Use `scipy.signal.stft` with:
  - `window='hann'` (Hann window, matches Go's gossp/stft default)
  - `nperseg=window` (window size)
  - `nfft=resolut` (FFT size)
  - `noverlap=window - (window // 4)` (75% overlap, typical for audio)

### ISTFT Configuration

- Use `scipy.signal.istft` with matching parameters
- Apply window normalization to ensure proper reconstruction

### Float16 Encoding

- Use `numpy.float16` for metadata values
- Pack as little-endian bytes using `struct.pack('<e', value)`
- Unpack using `struct.unpack('<e', bytes)`

### Performance Considerations

- Use vectorized NumPy operations where possible
- Avoid Python loops over large arrays
- Consider memory usage for long audio files (streaming not required for initial version)

## Future Enhancements

- **Streaming support**: Process very long audio files in chunks
- **Multi-channel audio**: Support stereo and multi-channel audio
- **Additional formats**: Support MP3, OGG, etc.
- **GPU acceleration**: Use CuPy for STFT on GPU
- **Compression**: Optional lossy compression for PNG spectrograms
