# PyPI Release Guide

This document describes how to release PocketLLM to PyPI.

## Prerequisites

1. **PyPI account and API token**
   - Create account at https://pypi.org/account/register/
   - Generate API token at https://pypi.org/manage/account/token/
   - Store token in `~/.pypirc`:
     ```ini
     [pypi]
     username = __token__
     password = pypi-AgE...your-token-here
     ```

2. **Install build tools**
   ```bash
   pip install --upgrade build twine
   ```

3. **Clean working directory**
   ```bash
   git status  # Should be clean
   git pull origin master
   ```

## Pre-release Checklist

- [ ] Update version in `pocketllm/__init__.py` and `pyproject.toml`
- [ ] Update `CHANGELOG.md` with release notes
- [ ] Run tests: `python -m pytest tests/test_install_smoke.py`
- [ ] Update README.md if needed
- [ ] Commit all changes: `git commit -m "Prepare v0.x.x release"`
- [ ] Create git tag: `git tag v0.x.x`

## Build the Distribution

### 1. Clean previous builds
```bash
rm -rf dist/ build/ *.egg-info pocketllm.egg-info
```

### 2. Build source distribution
```bash
python -m build --sdist --no-isolation
```

**Important:** Use `--no-isolation` because:
- `setup.py` needs the active environment's PyTorch
- CUDA extensions must compile against the host's CUDA toolkit
- An isolated build environment may have version mismatches

### 3. Verify the package
```bash
twine check dist/*
```

Expected output:
```
Checking dist/pocketllm-0.1.0.tar.gz: PASSED
```

### 4. Test installation locally (optional but recommended)
```bash
# Create a fresh virtual environment
python -m venv /tmp/test-pocketllm
source /tmp/test-pocketllm/bin/activate

# Install from the built package
pip install dist/pocketllm-0.1.0.tar.gz --no-build-isolation

# Run smoke test
python -c "import pocketllm; print(pocketllm.__version__)"

deactivate
```

## Upload to PyPI

### Test PyPI (recommended first)

1. Register at https://test.pypi.org/ (separate from production PyPI)

2. Upload to Test PyPI:
   ```bash
   twine upload --repository testpypi dist/*
   ```

3. Test installation from Test PyPI:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ \
       --extra-index-url https://pypi.org/simple/ \
       pocketllm
   ```

### Production PyPI

**⚠️ Warning: Once uploaded to PyPI, a version cannot be deleted or re-uploaded.**

1. Upload to production PyPI:
   ```bash
   twine upload dist/*
   ```

2. Verify at https://pypi.org/project/pocketllm/

3. Push git tags:
   ```bash
   git push origin master
   git push origin v0.x.x
   ```

## Post-release

1. **Create GitHub release**
   ```bash
   gh release create v0.x.x \
       --title "PocketLLM v0.x.x" \
       --notes-file CHANGELOG.md \
       dist/pocketllm-0.x.x.tar.gz
   ```

2. **Announce the release**
   - Update project README badges
   - Post on relevant forums/communities if appropriate

3. **Bump version for development**
   - Update version to next dev version (e.g., `0.2.0.dev0`)
   - Commit: `git commit -m "Bump version to 0.2.0.dev0"`

## Troubleshooting

### Build fails with "ModuleNotFoundError: No module named 'torch'"

**Solution:** Use `--no-build-isolation`:
```bash
python -m build --sdist --no-isolation
```

### CUDA extension compilation fails during `pip install`

**User should:**
1. Check CUDA toolkit is installed: `nvcc --version`
2. Install PyTorch first: `pip install torch`
3. Install with: `pip install pocketllm --no-build-isolation`

### C++ engine build fails

**Common causes:**
- Missing CMake: `pip install cmake` or `apt install cmake`
- Missing pybind11: `pip install pybind11`
- Missing NCCL: Install from NVIDIA or your package manager

**If C++ engine is not needed:**
```bash
POCKETLLM_BUILD_CPP=0 pip install pocketllm --no-build-isolation
```

### Build takes too long

The full build (CUDA extensions + C++ engine) takes 5-15 minutes. This is normal for the first installation. Subsequent upgrades reuse cached builds when possible.

## Version Numbering

PocketLLM follows [Semantic Versioning](https://semver.org/):

- **Major** (x.0.0): Breaking API changes
- **Minor** (0.x.0): New features, backward compatible
- **Patch** (0.0.x): Bug fixes, backward compatible

For pre-releases:
- **Alpha**: `0.1.0a1` (early testing)
- **Beta**: `0.1.0b1` (feature complete, testing)
- **RC**: `0.1.0rc1` (release candidate)
- **Dev**: `0.1.0.dev0` (development version)

## Current Release Status

**Version:** 0.1.0  
**Status:** Ready for first PyPI release  
**License:** MIT  
**Python:** 3.10, 3.11, 3.12  
**Platforms:** Linux (CUDA), Linux (CPU-only)

## Notes

- **No pre-built wheels yet**: Users must compile extensions during installation
- **CUDA versions**: Extensions compile against user's CUDA toolkit (11.8, 12.1, 12.4 tested)
- **C++ engine**: Optional, requires `POCKETLLM_BUILD_CPP=1`
- **Platform support**: Linux only (Ubuntu 22.04 tested, others should work)
- **Windows/macOS**: Not tested, may need adjustments

Future improvements:
- [ ] Add `cibuildwheel` for pre-compiled wheels (multiple CUDA versions × Python versions)
- [ ] Add Windows support
- [ ] Add macOS CPU-only support
