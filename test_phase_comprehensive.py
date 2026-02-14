#!/usr/bin/env python
"""
Comprehensive validation script for the phase.py implementation.
This script tests all major functionality without using a formal test framework.
"""

import numpy as np
import tempfile
import os
import sys
from phase import (
    Phase, pad, is_padded, spectral_normalize, spectral_denormalize,
    shrink, grow, pack_float16_to_bytes, unpack_bytes_to_float64,
    save_image, load_image, save_wav, load_wav, load_wav_with_sr,
    load_flac, load_flac_with_sr
)

def test_phase_initialization():
    """Test Phase class initialization."""
    print("Testing Phase initialization...")
    
    # Test default initialization
    p = Phase()
    assert p.num_freqs == 768, "Default num_freqs should be 768"
    assert p.window == 1280, "Default window should be 1280"
    assert p.resolut == 4096, "Default resolut should be 4096"
    assert p.y_reverse == True, "Default y_reverse should be True"
    assert p.volume_boost == 0.0, "Default volume_boost should be 0.0"
    
    # Test sample rate mapping
    for sr, expected_freqs in [(8000, 768), (16000, 768), (48000, 768),
                                (11025, 836), (22050, 836), (44100, 836)]:
        p = Phase(sample_rate=sr)
        assert p.num_freqs == expected_freqs, f"Sample rate {sr} should map to {expected_freqs} freqs"
    
    # Test custom values
    p = Phase(sample_rate=48000, window=2560, resolut=8192, y_reverse=False, volume_boost=1.5)
    assert p.window == 2560
    assert p.resolut == 8192
    assert p.y_reverse == False
    assert p.volume_boost == 1.5
    
    print("✓ Phase initialization tests passed")

def test_utility_functions():
    """Test utility functions."""
    print("Testing utility functions...")
    
    # Test pad
    audio = np.random.randn(1000)
    padded = pad(audio, 1280)
    assert len(padded) >= len(audio), "Padded length should be >= original"
    
    # Test is_padded
    result = is_padded(1000, len(padded), 1280)
    assert result == True, "Should detect padding"
    
    # Test spectral normalize/denormalize round trip
    spec = np.abs(np.random.randn(100, 3)) + 0.1  # Ensure positive values
    normalized = spectral_normalize(spec)
    denormalized = spectral_denormalize(normalized)
    # Should be close to original (within numerical precision)
    assert np.allclose(spec, denormalized, rtol=1e-5), "Normalize/denormalize should be inverse operations"
    
    # Test shrink/grow
    spec = np.random.randn(2048 * 10, 3)
    shrunken = shrink(spec, 4096, 768)
    assert shrunken.shape == (7680, 3), f"Shrunken shape should be (7680, 3), got {shrunken.shape}"
    grown = grow(shrunken, 4096, 768)
    assert grown.shape == spec.shape, "Grown shape should match original"
    
    print("✓ Utility function tests passed")

def test_metadata_encoding():
    """Test float16 metadata encoding/decoding."""
    print("Testing metadata encoding...")
    
    test_values = [1.5, -2.3, 0.0, 100.5, -50.25, 768.0, 48000.0, 0.001]
    
    for val in test_values:
        packed = pack_float16_to_bytes(val)
        assert len(packed) == 2, "Packed bytes should be 2 bytes"
        unpacked = unpack_bytes_to_float64(packed)
        # Float16 has limited precision
        rel_error = abs(unpacked - val) / (abs(val) + 1e-10)
        assert rel_error < 0.01, f"Relative error too large for {val}: {rel_error}"
    
    print("✓ Metadata encoding tests passed")

def test_image_save_load():
    """Test PNG image save/load."""
    print("Testing image save/load...")
    
    num_freqs = 768
    time_frames = 10
    spectrogram = np.random.randn(time_frames * num_freqs, 3)
    samples_in_mel = 1.5
    sample_rate = 48000
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Test save and load
        save_image(tmp_path, spectrogram, num_freqs, samples_in_mel, sample_rate, y_reverse=True)
        loaded_spec, loaded_samples, loaded_sr = load_image(tmp_path, y_reverse=True)
        
        assert spectrogram.shape == loaded_spec.shape, "Shape should be preserved"
        assert abs(loaded_samples - samples_in_mel) < 0.01, "samples_in_mel should be preserved"
        assert loaded_sr == sample_rate, "Sample rate should be preserved"
        
        print("✓ Image save/load tests passed")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_audio_io():
    """Test audio file I/O."""
    print("Testing audio I/O...")
    
    sample_rate = 48000
    duration = 0.1
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t) * 0.5
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Test save and load
        save_wav(tmp_path, audio, sample_rate)
        loaded_audio = load_wav(tmp_path)
        loaded_audio2, loaded_sr = load_wav_with_sr(tmp_path)
        
        assert loaded_audio.shape == audio.shape, "Shape should be preserved"
        assert loaded_sr == sample_rate, "Sample rate should be preserved"
        
        # Test clipping
        audio_clipped = np.array([0.5, 1.5, -0.5, -1.5, 2.0, -2.0])
        save_wav(tmp_path, audio_clipped, sample_rate)
        loaded_clipped = load_wav(tmp_path)
        assert np.max(loaded_clipped) <= 1.0, "Max value should be clipped to 1.0"
        assert np.min(loaded_clipped) >= -1.0, "Min value should be clipped to -1.0"
        
        print("✓ Audio I/O tests passed")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_phase_conversion():
    """Test audio to spectrogram and back."""
    print("Testing phase conversion...")
    
    sample_rate = 48000
    duration = 0.1
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t) * 0.5
    
    p = Phase(sample_rate=sample_rate)
    
    # Test to_phase
    spec = p.to_phase(audio)
    assert spec.shape[1] == 3, "Spectrogram should have 3 channels"
    assert spec.shape[0] % p.num_freqs == 0, "Spectrogram length should be multiple of num_freqs"
    
    # Test from_phase
    reconstructed = p.from_phase(spec)
    assert reconstructed.ndim == 1, "Reconstructed audio should be 1D"
    
    print("✓ Phase conversion tests passed")

def test_volume_boost():
    """Test volume boost functionality."""
    print("Testing volume boost...")
    
    sample_rate = 48000
    duration = 0.1
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t) * 0.5
    
    p1 = Phase(sample_rate=sample_rate, volume_boost=0.0)
    p2 = Phase(sample_rate=sample_rate, volume_boost=2.0)
    
    spec = p1.to_phase(audio)
    reconstructed1 = p1.from_phase(spec)
    reconstructed2 = p2.from_phase(spec)
    
    max1 = np.max(np.abs(reconstructed1))
    max2 = np.max(np.abs(reconstructed2))
    ratio = max2 / max1
    
    assert abs(ratio - 2.0) < 0.1, f"Volume boost should scale by 2x, got {ratio}"
    
    print("✓ Volume boost tests passed")

def test_high_level_workflow():
    """Test high-level convenience methods."""
    print("Testing high-level workflow...")
    
    sample_rate = 48000
    duration = 0.1
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t) * 0.5
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
        wav_path = tmp_wav.name
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_png:
        png_path = tmp_png.name
    with tempfile.NamedTemporaryFile(suffix='_out.wav', delete=False) as tmp_out:
        out_path = tmp_out.name
    
    try:
        # Save original audio
        save_wav(wav_path, audio, sample_rate)
        
        # Create Phase instance
        p = Phase(sample_rate=sample_rate)
        
        # Test to_phase_wav (WAV -> PNG)
        p.to_phase_wav(wav_path, png_path)
        assert os.path.exists(png_path), "PNG file should be created"
        
        # Test to_wav_png (PNG -> WAV)
        p.to_wav_png(png_path, out_path)
        assert os.path.exists(out_path), "Output WAV file should be created"
        
        # Load reconstructed audio
        reconstructed = load_wav(out_path)
        assert reconstructed.shape[0] > 0, "Reconstructed audio should not be empty"
        
        print("✓ High-level workflow tests passed")
    finally:
        for path in [wav_path, png_path, out_path]:
            if os.path.exists(path):
                os.remove(path)

def test_error_handling():
    """Test error handling."""
    print("Testing error handling...")
    
    # Test unsupported sample rate
    try:
        p = Phase(sample_rate=32000)
        print("✗ Should have raised ValueError for unsupported sample rate")
        sys.exit(1)
    except ValueError:
        pass  # Expected
    
    # Test loading non-existent file
    try:
        audio = load_wav('nonexistent_file.wav')
        print("✗ Should have raised exception for non-existent file")
        sys.exit(1)
    except Exception:
        pass  # Expected
    
    print("✓ Error handling tests passed")

def main():
    """Run all tests."""
    print("=" * 60)
    print("Running comprehensive validation tests for phase.py")
    print("=" * 60)
    print()
    
    try:
        test_phase_initialization()
        test_utility_functions()
        test_metadata_encoding()
        test_image_save_load()
        test_audio_io()
        test_phase_conversion()
        test_volume_boost()
        test_high_level_workflow()
        test_error_handling()
        
        print()
        print("=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 60)
        return 1
    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
