# Quick Start: Publishing to PyPI

Your package is ready to publish! Here's what to do:

## ✅ Package Built Successfully

The package has been built and is ready in the `dist/` directory:
- `phase_spectrogram-0.0.1-py3-none-any.whl` (wheel)
- `phase_spectrogram-0.0.1.tar.gz` (source distribution)

## 🔧 Before Publishing - Update These Files

### 1. Update `setup.py` (lines 18-21):
```python
author='Your Name',              # ← Change this
author_email='your.email@example.com',  # ← Change this
url='https://github.com/yourusername/phase-spectrogram',  # ← Change this
```

### 2. Update `pyproject.toml` (lines 11-12):
```toml
authors = [
    {name = "Your Name", email = "your.email@example.com"}  # ← Change this
]
```

### 3. Update URLs in `pyproject.toml` (lines 35-37):
```toml
[project.urls]
Homepage = "https://github.com/yourusername/phase-spectrogram"  # ← Change this
Repository = "https://github.com/yourusername/phase-spectrogram"  # ← Change this
"Bug Tracker" = "https://github.com/yourusername/phase-spectrogram/issues"  # ← Change this
```

## 📦 Publishing Steps

### Step 1: Create PyPI Account
1. Go to https://pypi.org/account/register/
2. Create account and verify email
3. (Optional) Create TestPyPI account at https://test.pypi.org/account/register/

### Step 2: Check Package Name Availability
Visit: https://pypi.org/project/phase-spectrogram/

If the name is taken, choose a different name and update it in:
- `setup.py` (line 16: `name='phase-spectrogram'`)
- `pyproject.toml` (line 5: `name = "phase-spectrogram"`)

### Step 3: Rebuild After Changes
```bash
# Clean old builds
rm -rf build/ dist/ *.egg-info/

# Rebuild
python -m build
```

### Step 4: Test on TestPyPI (Recommended)
```bash
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ --no-deps phase-spectrogram
```

### Step 5: Publish to PyPI
```bash
# Upload to PyPI
python -m twine upload dist/*
```

You'll be prompted for username and password.

### Step 6: Verify
```bash
# Install from PyPI
pip install phase-spectrogram

# Test it works
python -c "from phase import Phase; print('Success!')"
```

## 🎉 After Publishing

Users can now install your package with:
```bash
pip install phase-spectrogram
```

And use it like:
```python
from phase import Phase

phase = Phase(sample_rate=44100)
phase.to_phase_wav('input.wav', 'output.png')
phase.to_wav_png('output.png', 'reconstructed.wav')
```

## 📚 More Information

- **Full Guide**: See `PYPI_PUBLISHING_GUIDE.md`
- **Checklist**: See `CHECKLIST.md`
- **Package Summary**: See `PACKAGE_SUMMARY.md`
- **Examples**: Run `python example_usage.py`

## 🔄 Updating the Package

When you need to release a new version:

1. Update version in `__init__.py`:
   ```python
   __version__ = '0.0.2'  # Increment
   ```

2. Rebuild and upload:
   ```bash
   rm -rf build/ dist/ *.egg-info/
   python -m build
   python -m twine upload dist/*
   ```

## ⚠️ Important Notes

- You cannot upload the same version twice to PyPI
- Package names are case-insensitive and treat `-` and `_` as equivalent
- Once published, you cannot delete a version (only "yank" it)
- Keep your PyPI credentials secure

## 🆘 Need Help?

- Check `PYPI_PUBLISHING_GUIDE.md` for detailed troubleshooting
- Visit https://packaging.python.org/ for official documentation
- Check https://pypi.org/help/ for PyPI-specific help
