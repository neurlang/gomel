// Command tophase converts audio files (WAV/FLAC) to phase-preserving spectrogram images (PNG).
//
// This tool generates phase-preserving spectrograms from audio files and saves them as PNG images.
// Unlike mel spectrograms, phase spectrograms retain both magnitude and phase information,
// enabling high-quality audio reconstruction without iterative algorithms.
//
// Usage:
//
//	tophase <audio_file>
//
// The output PNG file will be named <audio_file>.png
//
// Supported input formats: .wav, .flac
package main
