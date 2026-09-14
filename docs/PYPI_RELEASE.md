# PyPI Release Guide

This is the single source of truth for releasing PocketLLM to PyPI. Any other release note that
disagrees with this file is out of date.

## Prerequisites

### 1. PyPI account and API tokens

Production and Test PyPI are separate services with separate accounts:

- Production: https://pypi.org/account/register/ and https://pypi.org/manage/account/token/
- Test: https://test.pypi.org/account/register/ and https://test.pypi.org/manage/account/token/

Create a token for each. The first publish for a project has to use an account-scoped token, because
a project-scoped token cannot be created until the project exists.

### 2. `~/.pypirc`

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-AgE...your-production-token-here

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-AgE...your-test-token-here
```

```bash
chmod 600 ~/.pypirc
```

### 3. Build tools

```bash
pip install --upgrade build twine
```

### 4. Clean working directory

```bash
git status            # should be clean apart from the release commit itself
git pull origin master
```

## Pre-release checklist

- [ ] Bump the version in **both** `pyproject.toml` and `pocketllm/__init__.py`
- [ ] Update `CHANGELOG.md` with the release notes
- [ ] Update `README.md` if the release changes installation or requirements
- [ ] Run the tests: `python -m pytest tests/test_install_smoke.py tests/test_native_build_preflight.py`
- [ ] Confirm the two version strings agree:
      `python -c "import pocketllm, tomllib, pathlib; print(pocketllm.__version__, tomllib.loads(pathlib.Path('pyproject.toml').read_text())['project']['version'])"`
- [ ] Commit all changes: `git commit -m "Release v0.x.x"`
- [ ] Create the tag: `git tag v0.x.x`

## Build the distribution

### 1. Clean previous builds

```bash
rm -rf dist/ build/ *.egg-info pocketllm.egg-info
```

### 2. Build the source distribution

```bash
python -m build --sdist --no-isolation
```

**`--no-isolation` is required:**
- `setup.py` needs the active environment's PyTorch
- CUDA extensions must compile against the host's CUDA toolkit
- An isolated build environment resolves a Torch that may not match that toolkit

### 3. Check the package

```bash
twine check dist/*
```

```
Checking dist/pocketllm-0.x.x.tar.gz: PASSED
```

### 4. Install the built sdist locally

```bash
python -m venv /tmp/test-pocketllm
source /tmp/test-pocketllm/bin/activate

pip install dist/pocketllm-0.x.x.tar.gz --no-build-isolation

python -c "import pocketllm; print(pocketllm.__version__)"
python tests/test_install_smoke.py

deactivate
```

This is the only step that proves the artifact installs. It compiles CUDA extensions and the native
C++ engine, so it needs a machine with a CUDA toolkit — a CI runner without one cannot do it.

## Upload

### Step 1: Test PyPI

Uploading here is always the first upload of a release, without exception.

```bash
twine upload --repository testpypi dist/*
```

### Step 2: Verify what Test PyPI actually serves

Do not stop at a successful upload. The 0.1.0 release uploaded cleanly, passed `twine check`, and
still shipped a distribution whose metadata claimed MIT while its rendered description said PolyForm
Noncommercial — because the description is `README.md`, and `twine check` does not read it.

Query the index for the metadata it is serving:

```bash
python - <<'PY'
import json, urllib.request
version = "0.x.x"  # the version just uploaded
with urllib.request.urlopen(f"https://test.pypi.org/pypi/pocketllm/{version}/json") as r:
    info = json.load(r)["info"]
print("version:   ", info["version"])
print("license:   ", info["license"])
print("rendered description mentions PolyForm:", "PolyForm" in (info.get("description") or ""))
PY
```

Expected: the version just uploaded, `license: MIT`, and `False`. The same check runs automatically as
the "Verify the published metadata" step of the publish workflow.

### Step 3: Install from Test PyPI

```bash
python -m venv /tmp/test-pocketllm
source /tmp/test-pocketllm/bin/activate

pip install torch                      # Test PyPI does not carry it
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    pocketllm --no-build-isolation

python -c "import pocketllm; print(pocketllm.__version__)"
python -c "import pocketllm_cpp; print('C++ engine: OK')"
python tests/test_install_smoke.py

deactivate
```

### Step 4: Production PyPI

**Once uploaded, a version cannot be deleted or replaced.** A mistake there is recoverable only by
yanking the release and publishing the next patch version.

```bash
twine upload dist/*
```

Verify at https://pypi.org/project/pocketllm/ and re-run the metadata query above against
`https://pypi.org/pypi/pocketllm/0.x.x/json`.

## Automated publishing (GitHub Actions)

`.github/workflows/publish-pypi.yml` runs the same sequence on a runner: build the sdist,
`twine check`, upload to Test PyPI, verify the metadata Test PyPI serves, and only then optionally
upload to production.

It is `workflow_dispatch`-only and the production upload is gated on the `publish_production` input,
which defaults to `no`. Leave it at `no` for a release candidate, inspect the verification step, and
re-run with `yes` once the metadata is known good.

Repository secrets the workflow needs:

| Secret | Purpose |
| --- | --- |
| `TEST_PYPI_API_TOKEN` | Upload to Test PyPI |
| `PYPI_API_TOKEN` | Upload to production PyPI |

Until both exist the workflow fails at its upload step. The workflow cannot install-test the
artifact: GitHub-hosted runners have no CUDA toolkit, and the sdist compiles CUDA extensions, so
installation verification stays a local step.

## Post-release

1. **Push the tag and create the GitHub release**

   ```bash
   git push origin master
   git push origin v0.x.x

   gh release create v0.x.x \
       --title "PocketLLM v0.x.x" \
       --notes-file CHANGELOG.md \
       dist/pocketllm-0.x.x.tar.gz
   ```

2. **Bump the version for development**

   Set the next development version in `pyproject.toml` and `pocketllm/__init__.py`, then commit:

   ```bash
   git commit -m "Bump version to 0.x.y.dev0"
   ```

   Skipping this makes the next release indistinguishable from the one just published.

## Troubleshooting

### `ModuleNotFoundError: No module named 'torch'` during the build

Use `--no-build-isolation`:

```bash
python -m build --sdist --no-isolation
```

### CUDA extension compilation fails during `pip install`

1. Check the toolkit: `nvcc --version`
2. Install PyTorch first: `pip install torch`
3. Install with `pip install pocketllm --no-build-isolation`

### Native C++ engine build fails

The build stops deliberately rather than continuing without `pocketllm_cpp`, because a silently
missing native module lets `backend="auto"` fall back to Torch kernels with no error to explain the
change in behaviour. The failure message names the missing prerequisite.

To install the PyTorch backend only:

```bash
POCKETLLM_BUILD_CPP=0 pip install pocketllm --no-build-isolation
```

### Build takes too long

The full build (CUDA extensions plus C++ engine) takes 5-15 minutes on a first install. Subsequent
installs reuse cached builds where possible.

## Version numbering

PocketLLM follows [Semantic Versioning](https://semver.org/):

- **Major** (`x.0.0`): breaking API changes
- **Minor** (`0.x.0`): new features, backward compatible
- **Patch** (`0.0.x`): bug fixes, backward compatible

Pre-releases: `0.1.0a1` (alpha), `0.1.0b1` (beta), `0.1.0rc1` (release candidate), `0.1.0.dev0`
(development).

## Known limitations of the published artifacts

- **No pre-built wheels.** Users compile extensions during installation, so they need a CUDA toolkit
  and 5-15 minutes.
- **CUDA versions.** Extensions compile against the user's toolkit. 11.8, 12.1, and 12.4 are
  exercised; anything else is untested.
- **Platform.** Linux only (Ubuntu 22.04 tested). Windows and macOS are untested.

Future improvements:

- [ ] `cibuildwheel` for pre-built wheels (CUDA version × Python version matrix)
- [ ] Windows support
- [ ] macOS CPU-only support
