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
- [ ] Run the tests: `python -m pytest tests/test_install_smoke.py tests/test_native_build_preflight.py tests/test_sdist_native_sources.py`
      `test_sdist_native_sources.py` is the one that guards the packaging step rather than the code: it checks that every source `cpp_engine/CMakeLists.txt` declares is in the sdist.
- [ ] Confirm the two version strings agree:
      `python -c "import re, pathlib, pocketllm; print(pocketllm.__version__, re.search(r'^version = \"(.+)\"', pathlib.Path('pyproject.toml').read_text(), re.M).group(1))"`
      The regular expression is not stylistic: `tomllib` is 3.11+, while this package supports 3.10, so a checklist step that imports it cannot be run on the oldest supported interpreter. The publish workflow reads the version the same way.
      This one does belong in the checkout — it compares the two version strings the source declares, and both sides come from the tree. The install steps below are the ones that must not run here.
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
REPO=$PWD                      # run this from the repository root
rm -rf /tmp/test-pocketllm
python -m venv /tmp/test-pocketllm
source /tmp/test-pocketllm/bin/activate

# --no-build-isolation tells pip not to provision [build-system].requires, so the
# venv has to already have what the build imports. A new venv has none of these,
# and the two ways that shows up are under Troubleshooting.
python -m pip install --upgrade pip
python -m pip install "torch>=2.0,<2.7" "setuptools>=68" wheel ninja cmake pybind11

pip install dist/pocketllm-0.x.x.tar.gz --no-build-isolation

cd /tmp
python -c "import pocketllm; print(pocketllm.__version__, pocketllm.__file__)"
python -c "import pocketllm_cpp; print('C++ engine: OK', pocketllm_cpp.__file__)"
python "$REPO/tests/test_install_smoke.py"

deactivate
```

`cd /tmp` is not incidental. `python -c` puts the working directory on `sys.path`, so run from the
checkout the first line imports `pocketllm/` from the source tree and prints the version the tree
has — which is what the release is bumping, so it agrees with the artifact by construction and proves
nothing. Both lines print the module path for the same reason: it has to be under the virtualenv.
The smoke test is run by path rather than by name, which puts the test's own directory, not the
checkout, on `sys.path`.

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
REPO=$PWD                      # run this from the repository root
rm -rf /tmp/test-pocketllm
python -m venv /tmp/test-pocketllm
source /tmp/test-pocketllm/bin/activate

# Test PyPI does not carry Torch; range as declared in pyproject.toml. These are
# the build imports as well -- see step 4 for why the venv needs them first.
python -m pip install --upgrade pip
python -m pip install "torch>=2.0,<2.7" "setuptools>=68" wheel ninja cmake pybind11

pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    pocketllm --no-build-isolation

cd /tmp   # otherwise the checkout shadows the installed package; see step 4
python -c "import pocketllm; print(pocketllm.__version__, pocketllm.__file__)"
python -c "import pocketllm_cpp; print('C++ engine: OK', pocketllm_cpp.__file__)"
python "$REPO/tests/test_install_smoke.py"

deactivate
```

This is the step that catches an sdist that is missing files it needs to compile — the archive
resolves, downloads and installs, and the failure is at build time.

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

### The install verification stops at `invalid command 'bdist_wheel'`

The venv is too bare for `--no-build-isolation`, which is the expected state of a venv that has just
been created. `python -m venv` bootstraps `pip` and `setuptools` from the interpreter's bundled
`ensurepip`; on Python 3.10 with pip 22.3.1 that is setuptools 65.5.0, which predates the version
that ships `bdist_wheel` as a built-in command — before setuptools 70.1 the command comes from the
separate `wheel` distribution, which a venv does not install. Nothing is built before this fails.

```bash
python -m pip install --upgrade pip
python -m pip install "setuptools>=68" wheel
```

`pyproject.toml` declares both in `[build-system].requires` and `setuptools>=68` in
`[project].dependencies`, but pip builds the wheel before installing the project's runtime
dependencies, so those declarations cannot supply the build that needs them.

### The install verification stops at "pybind11 is not importable"

Same cause, one step later: `cmake`, `pybind11` and `ninja` are also build imports, and the venv has
none of them. The message comes from this package's own preflight, which is reporting the venv's
toolchain rather than anything about the archive — it is not a sign that the artifact is broken.
Install them and retry:

```bash
python -m pip install ninja cmake pybind11
```

A venv created with `--system-site-packages` hides both of these failures, because it inherits the
base interpreter's `setuptools` and toolchain. That makes it useless for this check: it will report
success for an artifact the documented procedure cannot install.

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
