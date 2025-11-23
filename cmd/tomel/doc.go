// Command tomel converts audio files (WAV/FLAC) to mel spectrogram images (PNG).
//
// This tool generates mel-frequency spectrograms from audio files and saves them as PNG images.
// The mel spectrogram is a time-frequency representation that uses a perceptually-motivated
// mel-frequency scale, commonly used in speech and audio processing applications.
//
// Usage:
//
//	tomel <audio_file>
//
// The output PNG file will be named <audio_file>.png
//
// Supported input formats: .wav, .flac
package main
