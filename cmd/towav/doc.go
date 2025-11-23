// Command towav converts mel spectrogram images (PNG) back to audio files (WAV).
//
// This tool reconstructs audio waveforms from mel spectrogram PNG images using the
// Griffin-Lim algorithm. Since mel spectrograms don't preserve phase information,
// the reconstruction uses an iterative phase estimation process.
//
// Usage:
//
//	towav <png_file> [sample_rate]
//
// The output WAV file will be named <png_file>.wav
// Optional sample_rate parameter (default: 44100 Hz)
package main
