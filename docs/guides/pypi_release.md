# PyPI Release Guide

This is the single source of truth for releasing PocketLLM to PyPI. Any other release note that
disagrees with this file is out of date.

## Prerequisites

### 1. PyPI account and API tokens

Production and Test PyPI are separate services with separate accounts:

- Production: <https://pypi.org/account/register/> and <https://pypi.org/manage/account/token/>
- Test: <https://test.pypi.org/account/register/> and
  <https://test.pypi.org/manage/account/token/>

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

That is the whole toolchain. **This package compiles nothing** — `pyproject.toml` declares no
extension modules, the kernel ABI is stdlib-only, and the base dependency is `numpy`. There is no
CUDA toolkit requirement for the build, for the install, or for the test.

### 4. Clean working directory

```bash
git status
git switch main && git pull origin main
```

## Pre-release checklist

- [ ] Bump the version in **both** `pyproject.toml` and `pocketllm/__init__.py`
- [ ] Update `README.md` if the release changes installation or the status table
- [ ] Run the tests: `python -m pytest tests/ -q`, then `python scripts/check_test_baseline.py`
- [ ] Confirm the two version strings agree:

      ```bash
      python -c "import re, pathlib, pocketllm; print(pocketllm.__version__, re.search(r'^version = \"(.+)\"', pathlib.Path('pyproject.toml').read_text(), re.M).group(1))"
      ```

      The regular expression is not stylistic: `tomllib` is 3.11+, while this package supports 3.10,
      so a checklist step that imports it cannot run on the oldest supported interpreter. The publish
      workflow reads the version the same way.

- [ ] Commit all changes, on a branch, and merge through a pull request
- [ ] Create the tag: `git tag v0.x.x && git push origin v0.x.x`

## Build the distribution

### 1. Clean previous builds

```bash
rm -rf dist/ build/ *.egg-info pocketllm.egg-info
```

### 2. Build the source distribution

```bash
python -m build --sdist
```

`--no-isolation` is not needed: the build backend's requirements (`setuptools>=68`, `wheel`) are
ordinary wheels, and the project's own dependency is numpy.

### 3. Check the package

```bash
twine check dist/*
```

```
Checking dist/pocketllm-0.x.x.tar.gz: PASSED
```

`twine check` validates the README's rendering, which is the field that once shipped a license
mismatch. It does not read the *metadata* — that is step 2 of the upload section.

### 4. Install the built sdist locally

This is the step that catches an sdist missing files, and for this package it matters more than it
looks: the vendored GGML header is non-`.py` payload that `find_packages` does not ship on its own,
and a missing one fails at the first codebook read rather than at install.

```bash
REPO=$PWD                      # run this from the repository root
rm -rf /tmp/test-pocketllm
python -m venv /tmp/test-pocketllm
source /tmp/test-pocketllm/bin/activate

python -m pip install --upgrade pip
pip install dist/pocketllm-0.x.x.tar.gz

cd /tmp   # otherwise the checkout shadows the installed package; see below
python -c "import pocketllm; print(pocketllm.__version__, pocketllm.__file__)"
python "$REPO/tests/test_install_smoke.py"

deactivate
```

`cd /tmp` is not incidental. `python -c` puts the working directory on `sys.path`, so run from the
checkout the first line imports `pocketllm/` from the source tree and prints the version the tree
has — which is what the release is bumping, so it agrees with the artifact by construction and proves
nothing. The line prints the module path for the same reason: it has to be under the virtualenv. The
smoke test is run by path rather than by name, which puts the test's own directory, not the checkout,
on `sys.path`.

If you want to check the vendored header specifically — this fails loudly if the non-`.py` payload did
not ship, which an import alone would not catch:

```bash
python -c "from pocketllm.quant.ggml_tables import header_path; print(header_path())"
```

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

Expected: the version just uploaded, `license: Apache-2.0`, and `False`. The same check runs
automatically as the "Verify the published metadata" step of the publish workflow.

### Step 3: Install from Test PyPI

```bash
REPO=$PWD
rm -rf /tmp/test-pocketllm
python -m venv /tmp/test-pocketllm
source /tmp/test-pocketllm/bin/activate

python -m pip install --upgrade pip
# Pin the exact version under test. Unpinned, both indexes are searched and pip
# takes the highest version it can see, so this can quietly install a different
# release than the one just uploaded -- see Troubleshooting.
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    "pocketllm==0.x.x"

cd /tmp   # otherwise the checkout shadows the installed package
python -c "import pocketllm; print(pocketllm.__version__, pocketllm.__file__)"
python "$REPO/tests/test_install_smoke.py"

deactivate
```

The version is pinned for a reason. `--extra-index-url` does not scope the requirement to Test PyPI:
pip searches both indexes and takes the highest version it can see, so a listing that does not mention
the version just uploaded resolves to whatever production PyPI has instead, without a word. Pinning
turns that into `No matching distribution found`, which is a failure you can act on.

### Step 4: Production PyPI

**Once uploaded, a version cannot be deleted or replaced.** A mistake there is recoverable only by
yanking the release and publishing the next patch version.

```bash
twine upload dist/*
```

Verify at <https://pypi.org/project/pocketllm/> and re-run the metadata query above against
`https://pypi.org/pypi/pocketllm/0.x.x/json`.

## Automated publishing (GitHub Actions)

`.github/workflows/publish-pypi.yml` runs the same sequence on a runner: build the sdist,
`twine check`, upload to Test PyPI, verify the metadata Test PyPI serves, and only then optionally
upload to production.

It is `workflow_dispatch`-only and the production upload is gated on the `publish_production` input,
which defaults to `no`. Leave it at `no` for a release candidate, inspect the verification step, and
re-run with `yes` once the metadata is known good.

It does not run the test suite. **No workflow does** — see the note in the [guides index](index.md).

Repository secrets the workflow needs:

| Secret | Purpose |
| --- | --- |
| `TEST_PYPI_API_TOKEN` | Upload to Test PyPI |
| `PYPI_API_TOKEN` | Upload to production PyPI |

Unlike the previous release line, the workflow *can* install-test the artifact end to end: the
package has no build step and its only dependency is numpy, so a plain `pip install` on the runner is
the whole check. Adding that step is a reasonable follow-up.

## Post-release

Do this only after the artifact is on production PyPI. The GitHub release links to a version that
must already exist, and it cannot be created from a commit that is not on `main` — if the release was
prepared on a branch, merge it first so the tag has somewhere to land.

1. **Push the tag and create the GitHub release**

   ```bash
   git push origin main
   git push origin v0.x.x

   gh release create v0.x.x \
       --title "PocketLLM v0.x.x" \
       --notes-file /tmp/notes-0.x.x.md \
       dist/pocketllm-0.x.x.tar.gz
   ```

2. **Bump the version for development**

   Set the next development version in `pyproject.toml` and `pocketllm/__init__.py`, then commit on a
   branch and merge. Skipping this makes the next release indistinguishable from the one just
   published.

## Troubleshooting

### The install verification stops at `invalid command 'bdist_wheel'`

The venv is too bare. `python -m venv` bootstraps `pip` and `setuptools` from the interpreter's
bundled `ensurepip`; on Python 3.10 with pip 22.3.1 that is setuptools 65.5.0, which predates the
version that ships `bdist_wheel` as a built-in command — before setuptools 70.1 the command comes
from the separate `wheel` distribution, which a venv does not install. Nothing is built before this
fails.

```bash
python -m pip install --upgrade pip
python -m pip install "setuptools>=68" wheel
```

Relevant only if you build with `--no-isolation`; the default isolated build provisions these itself.

### The install verification builds an older release than the one just uploaded

It fails somewhere that has nothing to do with this release, or installs successfully and then reports
the previous version.

pip caches the simple-index page. If `https://test.pypi.org/simple/pocketllm/` was fetched before this
release was uploaded, pip reuses that listing; it has no link for the new version, so the resolver
falls back to the version production PyPI is offering. The published index is correct — only the
cached copy is old. It is visible without installing anything:

```bash
pip index versions pocketllm --cache-dir /tmp/pipcache --index-url https://test.pypi.org/simple/
```

Add `--no-cache-dir` to the install command to bypass the cache as well.

### The vendored header is missing from the installed package

A `FileNotFoundError` from `pocketllm.quant.ggml_tables.header_path()` after a *clean* install usually
means the non-`.py` payload did not ship. Two declarations have to agree:

- `[tool.setuptools.package-data]` in `pyproject.toml` — `"pocketllm.loader.gguf" = ["vendor/*.h",
  "vendor/*.md"]`
- `recursive-include pocketllm/loader/gguf/vendor *.h *.md` in `MANIFEST.in`

`tests/test_package_boundaries.py` fails if a non-Python file appears in the package without being
listed, which catches the reverse mistake.

### The GitHub release notes contain the whole changelog

`gh release create --notes-file CHANGELOG.md` publishes the entire file. Pass the extracted section
instead, rather than the whole file or `--notes "$(cat CHANGELOG.md)"`.

## Version numbering

PocketLLM follows [Semantic Versioning](https://semver.org/):

- **Major** (`x.0.0`): breaking API changes
- **Minor** (`0.x.0`): new features, backward compatible
- **Patch** (`0.0.x`): bug fixes, backward compatible

Pre-releases: `0.2.0a1` (alpha), `0.2.0b1` (beta), `0.2.0rc1` (release candidate), `0.2.0.dev0`
(development).

## Known limitations of the published artifacts

- **Nothing runs a model.** No device backend implements a kernel, so an installed `pocketllm` can
  list devices, resolve ops and read a GGUF checkpoint, and cannot serve. Say so in the release notes
  rather than letting a version number imply otherwise.
- **No pre-built wheels.** The sdist is pure Python; a wheel would be justified only if a build step
  appeared, which is the thing this tree exists not to have.
- **Platform.** Nothing in the package is platform-specific, but only Linux x86_64 and aarch64 are
  exercised. macOS is untested; the `mps` backend needs Apple Silicon and torch.