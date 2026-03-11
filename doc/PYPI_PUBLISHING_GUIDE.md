# Publishing phase-spectrogram to PyPI

This guide walks you through the steps to publish the `phase-spectrogram` package to PyPI.

## Prerequisites

1. **Create PyPI Account**
   - Go to https://pypi.org/account/register/
   - Create an account and verify your email

2. **Create TestPyPI Account (Optional but Recommended)**
   - Go to https://test.pypi.org/account/register/
   - Create an account for testing before publishing to the real PyPI

3. **Install Required Tools**
   ```bash
   pip install --upgrade pip
   pip install --upgrade build twine
   ```

## Step 1: Update Package Metadata

Before publishing, update the following files with your information:

### `setup.py` and `pyproject.toml`
- Replace `Your Name` with your actual name
- Replace `your.email@example.com` with your email
- Replace `yourusername` in URLs with your GitHub username
- Update the `url` field with your actual repository URL

### `__init__.py`
- Update `__version__` if needed (currently `0.0.1`)

### `README.md`
- Consider updating the README to focus on the Python package
- Add installation instructions: `pip install phase-spectrogram`
- Add Python usage examples

## Step 2: Clean Previous Builds

```bash
# Remove old build artifacts
rm -rf build/ dist/ *.egg-info/
```

## Step 3: Build the Package

```bash
# Build source distribution and wheel
python -m build
```

This creates two files in the `dist/` directory:
- `phase-spectrogram-0.0.1.tar.gz` (source distribution)
- `phase_spectrogram-0.0.1-py3-none-any.whl` (wheel)

## Step 4: Test the Package Locally (Optional)

```bash
# Install locally in editable mode
pip install -e .

# Or install from the built wheel
pip install dist/phase_spectrogram-0.0.1-py3-none-any.whl

# Test the import
python -c "from phase import Phase; print('Success!')"
```

## Step 5: Upload to TestPyPI (Recommended First Step)

```bash
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*
```

You'll be prompted for your TestPyPI username and password.

**Test the installation from TestPyPI:**
```bash
pip install --index-url https://test.pypi.org/simple/ --no-deps phase-spectrogram
```

## Step 6: Upload to PyPI (Production)

Once you've verified everything works on TestPyPI:

```bash
# Upload to PyPI
python -m twine upload dist/*
```

You'll be prompted for your PyPI username and password.

## Step 7: Verify Installation

```bash
# Install from PyPI
pip install phase-spectrogram

# Test it
python -c "from phase import Phase; p = Phase(sample_rate=44100); print('Package installed successfully!')"
```

## Using API Tokens (Recommended)

Instead of using username/password, you can use API tokens:

1. **Create API Token on PyPI**
   - Go to https://pypi.org/manage/account/token/
   - Create a new API token
   - Save it securely (you'll only see it once)

2. **Create `.pypirc` file** in your home directory:
   ```ini
   [distutils]
   index-servers =
       pypi
       testpypi

   [pypi]
   username = __token__
   password = pypi-AgEIcHlwaS5vcmc...  # Your PyPI token

   [testpypi]
   username = __token__
   password = pypi-AgENdGVzdC5weXBpLm9yZw...  # Your TestPyPI token
   ```

3. **Upload without prompts:**
   ```bash
   python -m twine upload dist/*
   ```

## Updating the Package

When you need to release a new version:

1. Update the version number in `__init__.py`
2. Update `CHANGELOG.md` (if you have one)
3. Clean old builds: `rm -rf build/ dist/ *.egg-info/`
4. Build: `python -m build`
5. Upload: `python -m twine upload dist/*`

## Common Issues

### Issue: "File already exists"
- You cannot upload the same version twice
- Increment the version number in `__init__.py`

### Issue: "Invalid distribution"
- Make sure all required files are present
- Check that `setup.py` and `pyproject.toml` are valid

### Issue: "Package name already taken"
- Choose a different package name
- Update `name` in `setup.py` and `pyproject.toml`

## Package Name Considerations

The current package name is `phase-spectrogram`. If this name is already taken on PyPI, consider alternatives:
- `phase-audio`
- `audio-phase-spectrogram`
- `phasespec`
- `phase-encoder`

Check availability: https://pypi.org/project/phase-spectrogram/

## Additional Resources

- PyPI Documentation: https://packaging.python.org/
- Twine Documentation: https://twine.readthedocs.io/
- Python Packaging Guide: https://packaging.python.org/tutorials/packaging-projects/
