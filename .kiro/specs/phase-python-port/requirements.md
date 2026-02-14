# Requirements Document

## Introduction

This document specifies the requirements for porting the Go `phase` package to Python. The phase package implements a phase-preserving spectrogram encoder/decoder that converts audio waveforms to spectrograms and back to audio without loss of phase information, enabling high-quality audio reconstruction.

## Glossary

- **Phase System**: The Python implementation of the phase-preserving spectrogram encoder/decoder
- **STFT**: Short-Time Fourier Transform, a method for analyzing the frequency content of signals over time
- **ISTFT**: Inverse Short-Time Fourier Transform, reconstructs time-domain signal from frequency-domain representation
- **Spectrogram**: A visual representation of the spectrum of frequencies in a signal as they vary with time
- **Phase-Preserving Spectrogram**: A spectrogram that retains both magnitude and phase information for lossless reconstruction
- **NumFreqs**: The number of frequency bins to use in the spectrogram
- **Window**: The window size for STFT analysis (default: 1280)
- **Resolut**: The FFT resolution (default: 4096)
- **YReverse**: Boolean flag to reverse the Y-axis when saving/loading PNG images (default: True)
- **Sample Rate**: The number of audio samples per second (e.g., 44100 Hz, 48000 Hz)

## Requirements

### Requirement 1

**User Story:** As a developer, I want to create a Phase encoder/decoder instance with configurable parameters, so that I can customize the audio processing behavior.

#### Acceptance Criteria

1. WHEN a user creates a Phase instance THEN the Phase System SHALL initialize with default values: YReverse=True, Window=1280, Resolut=4096
2. WHEN a user provides a sample rate of 8000, 16000, or 48000 Hz THEN the Phase System SHALL set NumFreqs to 768
3. WHEN a user provides a sample rate of 11025, 22050, or 44100 Hz THEN the Phase System SHALL set NumFreqs to 836
4. WHEN a user sets custom values for Window, Resolut, NumFreqs, YReverse, SampleRate, or VolumeBoost THEN the Phase System SHALL use those values for processing
5. THE Phase System SHALL expose all configuration parameters as accessible attributes

### Requirement 2

**User Story:** As a developer, I want to convert audio waveforms to phase-preserving spectrograms, so that I can analyze and manipulate audio in the frequency domain.

#### Acceptance Criteria

1. WHEN a user calls to_phase with an audio buffer THEN the Phase System SHALL apply padding to ensure proper STFT processing
2. WHEN performing STFT THEN the Phase System SHALL use the configured Window and Resolut parameters
3. WHEN extracting phase information THEN the Phase System SHALL preserve both real and imaginary components from the complex spectrum
4. WHEN generating the output spectrogram THEN the Phase System SHALL produce a 3-channel representation (realn1, realm0, realm1)
5. WHEN the spectrogram is generated THEN the Phase System SHALL apply spectral normalization using log2 transformation
6. WHEN shrinking the spectrum THEN the Phase System SHALL reduce from Resolut/2 frequency bins to NumFreqs bins

### Requirement 3

**User Story:** As a developer, I want to reconstruct audio from phase-preserving spectrograms, so that I can synthesize audio from frequency-domain representations.

#### Acceptance Criteria

1. WHEN a user calls from_phase with a spectrogram THEN the Phase System SHALL apply spectral denormalization using exp2 transformation
2. WHEN growing the spectrum THEN the Phase System SHALL expand from NumFreqs bins to Resolut/2 frequency bins
3. WHEN reconstructing the complex spectrum THEN the Phase System SHALL correctly reconstruct complex values from the 3-channel representation
4. WHEN performing ISTFT THEN the Phase System SHALL apply proper windowing and overlap-add synthesis
5. WHEN VolumeBoost is set THEN the Phase System SHALL apply the volume multiplier to the output audio
6. THE Phase System SHALL return the reconstructed audio buffer as a numpy array

### Requirement 4

**User Story:** As a developer, I want to load audio files in WAV and FLAC formats, so that I can process various audio file types.

#### Acceptance Criteria

1. WHEN a user calls load_wav with a file path THEN the Phase System SHALL load the audio as mono float64 samples
2. WHEN a user calls load_flac with a file path THEN the Phase System SHALL load the audio as mono float64 samples
3. WHEN a user calls load_wav_with_sr THEN the Phase System SHALL return both the audio buffer and sample rate
4. WHEN a user calls load_flac_with_sr THEN the Phase System SHALL return both the audio buffer and sample rate
5. WHEN a file cannot be loaded THEN the Phase System SHALL raise an appropriate exception

### Requirement 5

**User Story:** As a developer, I want to save audio buffers as WAV files, so that I can export processed audio.

#### Acceptance Criteria

1. WHEN a user calls save_wav with a file path, audio buffer, and sample rate THEN the Phase System SHALL write a mono WAV file
2. WHEN writing WAV files THEN the Phase System SHALL use 16-bit PCM encoding
3. WHEN the audio buffer contains values outside [-1, 1] THEN the Phase System SHALL clip the values to prevent distortion
4. THE Phase System SHALL handle file I/O errors gracefully with appropriate exceptions

### Requirement 6

**User Story:** As a developer, I want to save spectrograms as PNG images with embedded metadata, so that I can visualize and store spectrograms for later reconstruction.

#### Acceptance Criteria

1. WHEN a user converts audio to PNG THEN the Phase System SHALL encode the 3-channel spectrogram as RGB pixel values
2. WHEN saving PNG images THEN the Phase System SHALL embed metadata in the first column blue channel using float16 encoding
3. WHEN embedding metadata THEN the Phase System SHALL store: max values for each channel, min values for each channel, samples_in_mel ratio, and sample rate
4. WHEN YReverse is True THEN the Phase System SHALL flip the Y-axis when saving the image
5. WHEN normalizing for PNG THEN the Phase System SHALL map spectrogram values to 0-255 range per channel

### Requirement 7

**User Story:** As a developer, I want to load spectrograms from PNG images with embedded metadata, so that I can reconstruct audio from saved spectrograms.

#### Acceptance Criteria

1. WHEN a user loads a PNG file THEN the Phase System SHALL decode RGB pixel values to 3-channel spectrogram data
2. WHEN loading PNG images THEN the Phase System SHALL extract metadata from the first column blue channel
3. WHEN extracting metadata THEN the Phase System SHALL decode float16 values for: max/min values per channel, samples_in_mel ratio, and sample rate
4. WHEN YReverse is True THEN the Phase System SHALL flip the Y-axis when loading the image
5. WHEN denormalizing from PNG THEN the Phase System SHALL map 0-255 pixel values back to original spectrogram range using stored metadata

### Requirement 8

**User Story:** As a developer, I want high-level convenience functions for common workflows, so that I can quickly convert between audio files and PNG spectrograms.

#### Acceptance Criteria

1. WHEN a user calls to_phase_wav THEN the Phase System SHALL load a WAV file, convert to spectrogram, and save as PNG
2. WHEN a user calls to_phase_flac THEN the Phase System SHALL load a FLAC file, convert to spectrogram, and save as PNG
3. WHEN a user calls to_wav_png THEN the Phase System SHALL load a PNG spectrogram, convert to audio, and save as WAV
4. WHEN converting to WAV from PNG THEN the Phase System SHALL use embedded sample rate if SampleRate is not set
5. WHEN the original audio was padded THEN the Phase System SHALL trim the reconstructed audio to the original length using embedded metadata
