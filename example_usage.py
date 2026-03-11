#!/usr/bin/env python
"""
Example usage of the phase-spectrogram package.

This demonstrates the basic functionality of converting audio to spectrograms
and back to audio using phase-preserving encoding.
"""

from phase import Phase
import numpy as np

def example_basic_usage():
    """Basic usage example."""
    print("=== Basic Usage Example ===\n")
    
    # Initialize Phase encoder/decoder with 44.1kHz sample rate
    phase = Phase(sample_rate=44100)
    
    print(f"Configured for {phase.sample_rate}Hz with {phase.num_freqs} frequency bins")
    print(f"Window size: {phase.window}, FFT resolution: {phase.resolut}\n")
    
    # Create a simple test signal (1 second of 440Hz sine wave)
    duration = 1.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t)
    
    print("Converting audio to phase spectrogram...")
    spectrogram = phase.to_phase(audio)
    print(f"Spectrogram shape: {spectrogram.shape}")
    
    print("\nReconstructing audio from spectrogram...")
    reconstructed = phase.from_phase(spectrogram)
    print(f"Reconstructed audio length: {len(reconstructed)} samples")
    
    # Calculate reconstruction error
    min_len = min(len(audio), len(reconstructed))
    error = np.mean(np.abs(audio[:min_len] - reconstructed[:min_len]))
    print(f"Mean absolute error: {error:.6f}\n")


def example_file_conversion():
    """Example of converting audio files to/from spectrograms."""
    print("=== File Conversion Example ===\n")
    
    phase = Phase(sample_rate=44100)
    
    # These would work with actual audio files:
    print("To convert WAV to PNG spectrogram:")
    print("  phase.to_phase_wav('input.wav', 'output.png')")
    
    print("\nTo convert FLAC to PNG spectrogram:")
    print("  phase.to_phase_flac('input.flac', 'output.png')")
    
    print("\nTo convert PNG spectrogram back to WAV:")
    print("  phase.to_wav_png('spectrogram.png', 'output.wav')")
    print()


def example_different_sample_rates():
    """Example showing different sample rate configurations."""
    print("=== Different Sample Rates ===\n")
    
    supported_rates = [8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000]
    
    for rate in supported_rates:
        phase = Phase(sample_rate=rate)
        print(f"{rate:5d} Hz -> {phase.num_freqs:4d} frequency bins")
    
    print()


def example_advanced_options():
    """Example showing advanced configuration options."""
    print("=== Advanced Options ===\n")
    
    # High Dynamic Range (16-bit per channel PNG)
    phase_hdr = Phase(sample_rate=48000, HDR=True)
    print(f"HDR mode: {phase_hdr.num_freqs} frequency bins (16-bit PNG)")
    
    # With volume boost for reconstruction
    phase_boost = Phase(sample_rate=44100, volume_boost=2.0)
    print(f"Volume boost: {phase_boost.volume_boost}x")
    
    # With inverse hyperbolic sine compression
    phase_ihs = Phase(sample_rate=44100, IHS=True)
    print(f"IHS compression passes: {phase_ihs.IHS}")
    
    # Custom window and resolution
    phase_custom = Phase(
        sample_rate=44100,
        window=2560,
        resolut=8192,
        y_reverse=False
    )
    print(f"Custom: window={phase_custom.window}, resolut={phase_custom.resolut}")
    print()


if __name__ == "__main__":
    print("Phase-Preserving Spectrogram Package Examples")
    print("=" * 50 + "\n")
    
    example_basic_usage()
    example_file_conversion()
    example_different_sample_rates()
    example_advanced_options()
    
    print("For more information, see the documentation or README.md")
