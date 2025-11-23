// Package phase provides phase-preserving spectrogram generation and audio synthesis.
//
// This package implements conversion between audio waveforms and phase-preserving spectrograms,
// which retain both magnitude and phase information for high-quality audio reconstruction. It supports:
//   - Converting WAV/FLAC audio files to phase spectrograms (saved as PNG images)
//   - Reconstructing audio from phase spectrograms without iterative algorithms
//   - Direct phase preservation for lossless audio reconstruction
//   - STFT-based analysis and synthesis with configurable parameters
package phase
