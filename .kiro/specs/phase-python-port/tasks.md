# Implementation Plan

- [x] 1. Set up Python project structure and dependencies
  - Create `phase.py` module file
  - Create `requirements.txt` with dependencies: numpy, scipy, soundfile, Pillow, hypothesis (for testing)
  - Create `__init__.py` if creating a package
  - _Requirements: All_

- [x] 2. Implement core utility functions
  - [x] 2.1 Implement pad() function for audio buffer padding
    - Calculate padding based on window size and minimum target size (15 * window)
    - Handle both cases: buffer >= min_target_size and buffer < min_target_size
    - _Requirements: 2.1_

  - [x] 2.2 Implement is_padded() function to detect if audio was padded
    - Check if padded length matches expected padding calculation
    - Used for trimming reconstructed audio to original length
    - _Requirements: 8.5_

  - [x] 2.3 Implement spectral_normalize() function
    - Apply epsilon (1e-10) to prevent log(0)
    - Apply log2 transformation to all 3 channels
    - _Requirements: 2.5_

  - [x] 2.4 Implement spectral_denormalize() function
    - Apply exp2 transformation to all 3 channels (inverse of normalize)
    - _Requirements: 3.1_

  - [x] 2.5 Implement shrink() function to reduce frequency bins
    - Reduce from resolut/2 bins to num_freqs bins
    - Preserve time-frequency structure
    - _Requirements: 2.6_

  - [x] 2.6 Implement grow() function to expand frequency bins
    - Expand from num_freqs bins to resolut/2 bins
    - Replicate last frequency bin to fill expanded space
    - _Requirements: 3.2_

  - [ ]* 2.7 Write property test for normalize/denormalize round trip
    - **Property: Spectral normalization round trip**
    - **Validates: Requirements 2.5, 3.1**

  - [ ]* 2.8 Write property test for shrink/grow round trip
    - **Property: Shrink then grow preserves structure**
    - **Validates: Requirements 2.6, 3.2**

- [ ] 3. Implement Phase class initialization
  - [ ] 3.1 Create Phase class with __init__ method
    - Set default values: window=1280, resolut=4096, y_reverse=True, volume_boost=0.0
    - Implement sample rate to num_freqs mapping (768 for 48kHz family, 836 for 44.1kHz family)
    - Store all configuration as instance attributes
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [ ]* 3.2 Write unit tests for Phase initialization
    - Test default values are set correctly
    - Test sample rate to num_freqs mapping for all specified rates
    - Test custom configuration values are stored
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [ ] 4. Implement audio file I/O functions
  - [ ] 4.1 Implement load_wav() function
    - Use soundfile to load WAV as mono float64
    - Convert stereo to mono by averaging channels if needed
    - _Requirements: 4.1_

  - [ ] 4.2 Implement load_flac() function
    - Use soundfile to load FLAC as mono float64
    - Convert stereo to mono by averaging channels if needed
    - _Requirements: 4.2_

  - [ ] 4.3 Implement load_wav_with_sr() function
    - Return tuple of (audio_buffer, sample_rate)
    - _Requirements: 4.3_

  - [ ] 4.4 Implement load_flac_with_sr() function
    - Return tuple of (audio_buffer, sample_rate)
    - _Requirements: 4.4_

  - [ ] 4.5 Implement save_wav() function
    - Use soundfile to save as 16-bit PCM mono WAV
    - Clip audio values to [-1, 1] range before saving
    - _Requirements: 5.1, 5.2, 5.3_

  - [ ]* 4.6 Write unit tests for audio I/O
    - Test loading WAV and FLAC files with known test files
    - Test save_wav creates valid files
    - Test error handling for missing files
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 5.4_

  - [ ]* 4.7 Write property test for WAV round trip
    - **Property 6: WAV save/load round trip preserves audio**
    - **Validates: Requirements 5.1, 5.2**

  - [ ]* 4.8 Write property test for audio clipping
    - **Property 7: Audio clipping prevents out-of-range values**
    - **Validates: Requirements 5.3**

- [ ] 5. Implement to_phase() method for audio to spectrogram conversion
  - [ ] 5.1 Implement STFT computation
    - Apply padding to audio buffer
    - Use scipy.signal.stft with Hann window, configured window size and FFT resolution
    - _Requirements: 2.1, 2.2_

  - [ ] 5.2 Extract 3-channel phase representation from STFT
    - For each frequency bin j: extract v0 = spectrum[j+1] and v1 = spectrum[resolut-j-1]
    - Store realn1 = imag(v0), realm0 = real(v1), realm1 = imag(v1)
    - Create output array with shape (time_frames * resolut/2, 3)
    - _Requirements: 2.3, 2.4_

  - [ ] 5.3 Apply shrink and normalization
    - Call shrink() to reduce to num_freqs bins
    - Call spectral_normalize() to apply log2 transform
    - _Requirements: 2.5, 2.6_

  - [ ]* 5.4 Write property test for spectrogram dimensions
    - **Property 4: Spectrogram output has correct dimensions**
    - **Validates: Requirements 2.4, 2.6**

- [ ] 6. Implement from_phase() method for spectrogram to audio conversion
  - [ ] 6.1 Implement denormalization and grow
    - Call spectral_denormalize() to apply exp2 transform
    - Call grow() to expand to resolut/2 bins
    - _Requirements: 3.1, 3.2_

  - [ ] 6.2 Reconstruct complex spectrum from 3-channel representation
    - For each frequency bin: reconstruct v0 = complex(realm0, realn1) and v1 = complex(realm0, realm1)
    - Place v0 at spectrum[j+1] and v1 at spectrum[resolut-j-1]
    - Create full complex spectrum array
    - _Requirements: 3.3_

  - [ ] 6.3 Implement ISTFT computation
    - Use scipy.signal.istft with matching parameters
    - Apply proper window normalization
    - _Requirements: 3.4_

  - [ ] 6.4 Apply volume boost if configured
    - Multiply output by volume_boost if volume_boost > 0
    - Return as numpy float64 array
    - _Requirements: 3.5, 3.6_

  - [ ]* 6.5 Write property test for audio round trip
    - **Property 1: Audio-to-spectrogram-to-audio round trip preserves signal**
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 3.1, 3.2, 3.3, 3.4**

  - [ ]* 6.6 Write property test for volume boost
    - **Property 5: Volume boost scales output**
    - **Validates: Requirements 3.5**

- [ ] 7. Implement PNG metadata encoding/decoding functions
  - [ ] 7.1 Implement pack_float16_to_bytes() function
    - Convert float64 to float16 using numpy
    - Pack as 2 bytes in little-endian format
    - _Requirements: 6.2_

  - [ ] 7.2 Implement unpack_bytes_to_float64() function
    - Unpack 2 bytes as little-endian float16
    - Convert to float64
    - _Requirements: 7.2_

  - [ ] 7.3 Implement metadata embedding in save_image()
    - Calculate max/min values for each of 3 channels
    - Pack metadata: [max0, max1, max2, min0, min1, min2, samples_in_mel, sample_rate]
    - Embed in first column (x=0) blue channel pixels 0-15
    - _Requirements: 6.2, 6.3_

  - [ ] 7.4 Implement metadata extraction in load_image()
    - Extract blue channel from first column pixels 0-15
    - Unpack 8 float16 values for metadata
    - Return metadata along with spectrogram
    - _Requirements: 7.2, 7.3_

  - [ ]* 7.5 Write property test for metadata round trip
    - **Property 9: Metadata embedding preserves reconstruction parameters**
    - **Validates: Requirements 6.2, 6.3, 7.2, 7.3**

- [ ] 8. Implement PNG image save/load functions
  - [ ] 8.1 Implement save_image() function
    - Calculate stride (time frames) from spectrogram length and num_freqs
    - Find max/min values per channel for normalization
    - Normalize each channel to 0-255 range
    - Create RGB image with PIL: R=channel0, G=channel1, B=channel2
    - Embed metadata in first column blue channel
    - Apply Y-axis flip if y_reverse=True
    - Save as PNG
    - _Requirements: 6.1, 6.4, 6.5_

  - [ ] 8.2 Implement load_image() function
    - Load PNG with PIL
    - Extract metadata from first column blue channel
    - Decode RGB pixels to 3-channel spectrogram
    - Denormalize from 0-255 to original range using metadata
    - Apply Y-axis flip if y_reverse=True
    - Return spectrogram, samples ratio, and sample rate
    - _Requirements: 7.1, 7.4, 7.5_

  - [ ]* 8.3 Write property test for PNG round trip
    - **Property 2: PNG round trip preserves spectrogram data**
    - **Validates: Requirements 6.1, 6.2, 6.3, 6.5, 7.1, 7.2, 7.3, 7.5**

  - [ ]* 8.4 Write property test for Y-axis reversal
    - **Property 8: Y-axis reversal is consistent**
    - **Validates: Requirements 6.4, 7.4**

- [ ] 9. Implement high-level convenience methods
  - [ ] 9.1 Implement to_phase_wav() method
    - Load WAV file using load_wav_with_sr()
    - Call to_phase() to generate spectrogram
    - Calculate samples_in_mel ratio
    - Call save_image() with spectrogram and metadata
    - _Requirements: 8.1_

  - [ ] 9.2 Implement to_phase_flac() method
    - Load FLAC file using load_flac_with_sr()
    - Call to_phase() to generate spectrogram
    - Calculate samples_in_mel ratio
    - Call save_image() with spectrogram and metadata
    - _Requirements: 8.2_

  - [ ] 9.3 Implement to_wav_png() method
    - Load PNG using load_image() to get spectrogram and metadata
    - Call from_phase() to reconstruct audio
    - Use embedded sample rate if self.sample_rate is not set
    - Trim padding if original length is known using is_padded()
    - Call save_wav() to write output file
    - _Requirements: 8.3, 8.4, 8.5_

  - [ ]* 9.4 Write integration tests for high-level workflows
    - Test to_phase_wav() with known WAV file
    - Test to_phase_flac() with known FLAC file
    - Test to_wav_png() with known PNG file
    - Test complete round trip: WAV → PNG → WAV
    - _Requirements: 8.1, 8.2, 8.3_

  - [ ]* 9.5 Write property test for padding removal
    - **Property 10: Padding is correctly applied and removed**
    - **Validates: Requirements 2.1, 8.5**

- [ ] 10. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
