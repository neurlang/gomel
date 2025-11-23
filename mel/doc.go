// Package mel provides mel-frequency spectrogram generation and audio synthesis.
//
// This package implements conversion between audio waveforms and mel-scale spectrograms,
// which are commonly used in audio processing and speech recognition. It supports:
//   - Converting WAV/FLAC audio files to mel spectrograms (saved as PNG images)
//   - Reconstructing audio from mel spectrograms using Griffin-Lim algorithm
//   - Configurable mel filterbank parameters (frequency range, number of mel bands)
//   - STFT-based analysis and synthesis with customizable window sizes
package mel
