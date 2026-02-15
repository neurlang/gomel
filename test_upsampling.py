#!/usr/bin/env python3
"""Test upsampling codec on various sample rate files."""

import os
import glob
from phase import Phase

# Directory containing test files
test_dir = "/tmp/upsampling"

# Find all FLAC files
flac_files = sorted(glob.glob(os.path.join(test_dir, "*.flac")))

print(f"Found {len(flac_files)} FLAC files to process\n")

for flac_path in flac_files:
    basename = os.path.splitext(os.path.basename(flac_path))[0]
    png_path = os.path.join(test_dir, f"{basename}.png")
    wav_path = os.path.join(test_dir, f"{basename}_recovered.wav")
    
    print(f"Processing: {basename}.flac")
    
    try:
        # Detect sample rate from filename or file
        import soundfile as sf
        _, sample_rate = sf.read(flac_path, frames=1)
        
        # Check if sample rate is supported
        supported_rates = [8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000]
        if sample_rate not in supported_rates:
            print(f"  ⚠ Skipping: {sample_rate} Hz not supported")
            print()
            continue
        
        print(f"  Sample rate: {sample_rate} Hz")
        
        # Step 1: FLAC -> PNG
        print(f"  Converting FLAC -> PNG...")
        phase_encoder = Phase(sample_rate=sample_rate)
        phase_encoder.to_phase_flac(flac_path, png_path)
        print(f"  ✓ Created: {basename}.png")
        
        # Step 2: PNG -> WAV
        print(f"  Converting PNG -> WAV...")
        phase_decoder = Phase(sample_rate=sample_rate)  # Use same sample_rate to get correct num_freqs
        phase_decoder.to_wav_png(png_path, wav_path)
        print(f"  ✓ Created: {basename}_recovered.wav")
        
        # Check file sizes
        png_size = os.path.getsize(png_path) / 1024
        wav_size = os.path.getsize(wav_path) / 1024
        print(f"  PNG size: {png_size:.1f} KB, WAV size: {wav_size:.1f} KB")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print()

print("Processing complete!")
