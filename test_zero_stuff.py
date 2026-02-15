#!/usr/bin/env python3
"""Test zero stuffing upsampling."""

import numpy as np
from phase import zero_stuff_upsample

# Test case 1: 22050 -> 44100 (2x upsampling)
print("Test 1: 22050 -> 44100 (zero_pad=1, zero_shift=1)")
audio_22k = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
upsampled = zero_stuff_upsample(audio_22k, zero_pad=1, zero_shift=1)
print(f"  Input:  {audio_22k}")
print(f"  Output: {upsampled}")
print(f"  Expected: [1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0]")
print(f"  Length: {len(audio_22k)} -> {len(upsampled)} (expected {len(audio_22k) * 2})")
print()

# Test case 2: 11025 -> 44100 (4x upsampling)
print("Test 2: 11025 -> 44100 (zero_pad=1, zero_shift=3)")
audio_11k = np.array([1.0, 2.0, 3.0])
upsampled = zero_stuff_upsample(audio_11k, zero_pad=1, zero_shift=3)
print(f"  Input:  {audio_11k}")
print(f"  Output: {upsampled}")
print(f"  Expected: [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0]")
print(f"  Length: {len(audio_11k)} -> {len(upsampled)} (expected {len(audio_11k) * 4})")
print()

# Test case 3: 8000 -> 48000 (6x upsampling)
print("Test 3: 8000 -> 48000 (zero_pad=1, zero_shift=5)")
audio_8k = np.array([1.0, 2.0])
upsampled = zero_stuff_upsample(audio_8k, zero_pad=1, zero_shift=5)
print(f"  Input:  {audio_8k}")
print(f"  Output: {upsampled}")
print(f"  Expected: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]")
print(f"  Length: {len(audio_8k)} -> {len(upsampled)} (expected {len(audio_8k) * 6})")
print()

# Test with actual audio file
print("Test 4: Real audio file (22050 Hz)")
import soundfile as sf
audio, sr = sf.read('/tmp/upsampling/22khz.flac', dtype='float64')
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)
print(f"  Original length: {len(audio)} samples at {sr} Hz")
upsampled = zero_stuff_upsample(audio, zero_pad=1, zero_shift=1)
print(f"  Upsampled length: {len(upsampled)} samples (expected {len(audio) * 2})")
print(f"  First 10 original: {audio[:5]}")
print(f"  First 10 upsampled: {upsampled[:10]}")
