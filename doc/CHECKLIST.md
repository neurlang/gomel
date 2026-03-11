# PyPI Publishing Checklist

Use this checklist before publishing to PyPI:

## Pre-Publishing

- [x] Update `setup.py` with your name, email, and repository URL
- [x] Update `pyproject.toml` with your name, email, and repository URL
- [x] Update `__init__.py` version number if needed
- [x] Verify LICENSE file exists and is appropriate
- [x] Update README.md with Python-specific installation and usage
- [ ] Test all functionality locally
- [ ] Run tests: `python -m pytest` (if you have tests)
- [ ] Check package name availability on PyPI

## Building

- [ ] Install build tools: `pip install --upgrade build twine`
- [ ] Clean old builds: `rm -rf build/ dist/ *.egg-info/`
- [ ] Build package: `python -m build`
- [ ] Verify dist/ contains both .tar.gz and .whl files

## Testing

- [ ] Install locally: `pip install -e .`
- [ ] Test import: `python -c "from phase import Phase; print('OK')"`
- [ ] Upload to TestPyPI: `python -m twine upload --repository testpypi dist/*`
- [ ] Install from TestPyPI and test

## Publishing

- [x] Create PyPI account at https://pypi.org/account/register/
- [ ] Upload to PyPI: `python -m twine upload dist/*`
- [ ] Verify package page on PyPI
- [ ] Install from PyPI: `pip install phase-spectrogram`
- [ ] Test installed package

## Post-Publishing

- [ ] Tag release in git: `git tag v0.0.1 && git push --tags`
- [ ] Update documentation with installation instructions
- [ ] Announce release (if applicable)

## Quick Commands

```bash
# Complete workflow
rm -rf build/ dist/ *.egg-info/
python -m build
python -m twine upload --repository testpypi dist/*  # Test first
python -m twine upload dist/*  # Then production
```
