// Command fromphase converts phase-preserving spectrogram images (PNG) back to audio files (WAV).
//
// This tool reconstructs audio waveforms from phase-preserving spectrogram PNG images.
// Since phase information is preserved in the spectrogram, the reconstruction is direct
// and produces high-quality audio output.
//
// Usage:
//
//	fromphase <png_file> [sample_rate]
//
// The output WAV file will be named <png_file>.wav
// Optional sample_rate parameter (default: 44100 Hz)
package main
