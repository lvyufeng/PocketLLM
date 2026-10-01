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

This release is pure Python, so there is no CUDA toolkit requirement for the build itself. The
runtime depends on `relic-core`, which is not on PyPI yet and is installed from its checkout:
`pip install -e ../relic-core --no-build-isolation`.

### 4. Clean working directory

```bash
git status            # should be clean apart from the release commit itself
git pull origin master
```

## Pre-release checklist

- [ ] Bump the version in **both** `pyproject.toml` and `pocketllm/__init__.py`
- [ ] Update `CHANGELOG.md` with the release notes
- [ ] Update `README.md` if the release changes installation or requirements
- [ ] Run the tests: `python -m pytest tests/test_install_smoke.py`
      The sdist is pure Python now, so there is no native-source packaging check to run: the
      `cpp_engine` CMake sources and the `src/csrc/` sources this project used to ship are in
      [relic-engine](https://github.com/lvyufeng/relic-engine) and
      [relic-core](https://github.com/lvyufeng/relic-core) respectively.
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
- `setup.py` needs the active environment's PyTorch (it is a runtime dependency, declared in
  `pyproject.toml`)
- An isolated build environment would resolve its own setuptools and wheel rather than using the
  ones the release environment already has

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
python -m pip install "torch>=2.0,<2.7" "setuptools>=68" wheel

# The runtime's native operator library, not on PyPI: install it from its
# checkout first, or resolving the dependency below cannot fetch it.
pip install -e "$REPO/../relic-core" --no-build-isolation

pip install dist/pocketllm-0.x.x.tar.gz --no-build-isolation

cd /tmp
python -c "import pocketllm; print(pocketllm.__version__, pocketllm.__file__)"
python "$REPO/tests/test_install_smoke.py"

deactivate
```

This is the step that catches an sdist that is missing files. The archive is pure Python, so the
failure mode is an import error rather than a compile error, and the install no longer needs a CUDA
toolkit.

`cd /tmp` is not incidental. `python -c` puts the working directory on `sys.path`, so run from the
checkout the first line imports `pocketllm/` from the source tree and prints the version the tree
has — which is what the release is bumping, so it agrees with the artifact by construction and proves
nothing. The line prints the module path for the same reason: it has to be under the virtualenv.
The smoke test is run by path rather than by name, which puts the test's own directory, not the
checkout, on `sys.path`.

This is the only step that proves the artifact installs. The build is pure Python now — the CUDA
kernels compile in relic-core, its own package — so a machine with a CUDA toolkit is needed for the
`relic-core` install, not for this one.

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
python -m pip install "torch>=2.0,<2.7" "setuptools>=68" wheel
# relic-core is not on PyPI; install it from its checkout so the dependency below
# can resolve. See step 4.
pip install -e "$REPO/../relic-core" --no-build-isolation

# Pin the exact version under test. Unpinned, both indexes are searched and pip
# takes the highest version it can see, so this can quietly install a different
# release than the one just uploaded -- see Troubleshooting.
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    "pocketllm==0.x.x" --no-build-isolation

cd /tmp   # otherwise the checkout shadows the installed package; see step 4
python -c "import pocketllm; print(pocketllm.__version__, pocketllm.__file__)"
python "$REPO/tests/test_install_smoke.py"

deactivate
```

This is the step that catches an sdist that is missing files it needs — the archive resolves,
downloads and installs, and the failure is at import time.

The version is pinned for a reason. `--extra-index-url` does not scope the requirement to Test PyPI:
pip searches both indexes and takes the highest version it can see, so a listing that does not
mention the version just uploaded resolves to whatever production PyPI has instead, without a word.
Pinning turns that into `No matching distribution found`, which is a failure you can act on. See
Troubleshooting for the way this actually happens.

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
artifact end to end: `relic-core` is not on PyPI and a runner would have to build it from source,
so installation verification stays a local step.

## Post-release

Do this only after the artifact is on production PyPI. The GitHub release links to a version that
must already exist, and it cannot be created from a commit that is not on `master` — if the release
was prepared on a branch, merged first and pull `master` so the tag has somewhere to land.

1. **Push the tag and create the GitHub release**

   ```bash
   git push origin master
   git push origin v0.x.x

   # Release notes are this version's changelog section, not the whole file. Passing
   # CHANGELOG.md directly publishes every previous release's notes as well; see
   # Troubleshooting.
   VERSION=0.x.x
   awk -v hdr="## [$VERSION]" '
       index($0, hdr) == 1 { found = 1 }
       found && /^## \[/ && index($0, hdr) != 1 { exit }
       found
   ' CHANGELOG.md > /tmp/notes-$VERSION.md

   gh release create v$VERSION \
       --title "PocketLLM v$VERSION" \
       --notes-file /tmp/notes-$VERSION.md \
       dist/pocketllm-$VERSION.tar.gz
   ```

   The comparison is on the bracketed header rather than a regular expression on purpose: an
   unescaped `[0.x.x]` is a character class, and `index` on `## [0.1.1]` also declines to match a
   `## [0.1.10]` header.

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

### The install verification cannot resolve `relic-core`

The venv does not have the operator library, which is a runtime dependency and is not on PyPI.
Install it from its checkout before installing the archive:

```bash
pip install -e ../relic-core --no-build-isolation
```

A venv created with `--system-site-packages` hides this failure, because it inherits the base
interpreter's installed `relic-core`. That makes it useless for this check: it will report success
for an artifact the documented procedure cannot install.

### The install verification builds an older release than the one just uploaded

The step fails somewhere that has nothing to do with this release — a missing file, or an installed
distribution that is not the one just uploaded — or installs successfully and then reports the
previous version.

pip caches the simple-index page. If `https://test.pypi.org/simple/pocketllm/` was fetched before
this release was uploaded, pip reuses that listing; it has no link for the new version, so the
resolver falls back to the version production PyPI is offering. The published index is correct — only
the cached copy is old. It is visible without installing anything:

```bash
pip index versions pocketllm --index-url https://test.pypi.org/simple/
```

If that stops at the previous release while `https://test.pypi.org/pypi/pocketllm/<version>/json`
answers for the new one, this is what happened. Passing a fresh cache directory forces a real fetch:

```bash
pip index versions pocketllm --cache-dir /tmp/pipcache --index-url https://test.pypi.org/simple/
```

The version pin in step 3 is what keeps this from being silent: with it, a listing that does not
carry the release under test fails with `No matching distribution found` rather than installing
something else. Add `--no-cache-dir` to the install command to bypass the cache as well.

### The GitHub release notes contain the whole changelog

`gh release create --notes-file CHANGELOG.md` publishes the entire file. The v0.1.1 release notes
built that way would be 216 lines: the 0.1.1 section, then the 0.1.0 section with its feature list
and performance table, then the link definitions. A reader looking for what changed in this release
has to find it.

Pass the extracted section instead, as the post-release step does. The same applies to
`--notes "$(cat CHANGELOG.md)"`.

### `ModuleNotFoundError: No module named 'torch'` during the build

Use `--no-build-isolation`:

```bash
python -m build --sdist --no-isolation
```

### The `relic-core` kernel build fails during its install

`pocketllm` itself does not compile, but its operator library does, so the toolkit matters for that
step:

1. Check the toolkit: `nvcc --version`
2. Install PyTorch first: `pip install torch`
3. Install relic-core with `pip install -e ../relic-core --no-build-isolation`

### Build takes too long

`pocketllm` itself is pure Python and installs in seconds. The only long step is `relic-core`, which
compiles the native kernels and takes several minutes on a first install; subsequent installs reuse
cached builds where possible.

## Version numbering

PocketLLM follows [Semantic Versioning](https://semver.org/):

- **Major** (`x.0.0`): breaking API changes
- **Minor** (`0.x.0`): new features, backward compatible
- **Patch** (`0.0.x`): bug fixes, backward compatible

Pre-releases: `0.1.0a1` (alpha), `0.1.0b1` (beta), `0.1.0rc1` (release candidate), `0.1.0.dev0`
(development).

## Known limitations of the published artifacts

- **No pre-built wheels.** The sdist is pure Python, but its `relic-core` dependency is not on PyPI:
  users build it from its checkout, so they need a CUDA toolkit and the time that compile takes.
- **CUDA versions.** The `relic-core` kernels compile against the user's toolkit. 11.8, 12.1, and
  12.4 are exercised; anything else is untested.
- **Platform.** Linux only (Ubuntu 22.04 tested). Windows and macOS are untested.

Future improvements:

- [ ] `cibuildwheel` for pre-built wheels (CUDA version × Python version matrix)
- [ ] Windows support
- [ ] macOS CPU-only support
