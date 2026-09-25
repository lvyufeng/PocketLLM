"""The ``prism.hadamard.*`` block of the released checkpoint, for the tests that read it.

``tests/data/ternary_bonsai_hadamard.json`` is the block as it is stored in
``prism-ml/Ternary-Bonsai-2-27B-gguf``'s ``Ternary-Bonsai-2-27B-PTQ1_0.gguf``, so a
test that only needs the metadata runs without the 5.5 GiB checkpoint.  The sign
vector is the file's own, bit-packed; the widths are 5120, 6144 and 17408 and the
concatenation of the three is what ``signs_for_width`` slices.

Shared by ``test_prism_hadamard.py`` (the contract) and
``test_prism_hadamard_transform.py`` (the computation), because the second is only
meaningful on the first's numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.loader.gguf.prism_hadamard import HadamardSpec, parse_hadamard_spec

FIXTURE_PATH = Path(__file__).parent / "data" / "ternary_bonsai_hadamard.json"


def load_hadamard_fixture() -> dict:
    with FIXTURE_PATH.open() as handle:
        return json.load(handle)


def unpack_signs(blob: str, count: int) -> list[int]:
    """Expand the fixture's bit-packed sign vector: one bit per entry, +1 when set."""
    raw = bytes.fromhex(blob)
    assert len(raw) * 8 >= count
    return [1 if raw[i >> 3] >> (i & 7) & 1 else -1 for i in range(count)]


def hadamard_metadata(raw: dict | None = None, **overrides) -> dict:
    """The fixture as GGUF metadata, with per-key overrides for the refusal tests.

    Passing a key as ``None`` removes it, which is how the missing-key cases are
    stated.
    """
    fixture = load_hadamard_fixture() if raw is None else dict(raw)
    metadata = {
        "prism.hadamard.version": fixture["version"],
        "prism.hadamard.transform": fixture["transform"],
        "prism.hadamard.axis": fixture["axis"],
        "prism.hadamard.block_size": fixture["block_size"],
        "prism.hadamard.sign_mode": fixture["sign_mode"],
        "prism.hadamard.gdn_v_grouped": fixture["gdn_v_grouped"],
        "prism.hadamard.sign_widths": list(fixture["sign_widths"]),
        "prism.hadamard.sign_values": unpack_signs(
            fixture["sign_values_bits"], sum(fixture["sign_widths"])
        ),
        "prism.hadamard.weight_names": list(fixture["weight_names"]),
        "prism.hadamard.inverse_weight_names": list(fixture["inverse_weight_names"]),
    }
    for key, value in overrides.items():
        if value is None:
            metadata.pop(key, None)
        else:
            metadata[key] = value
    return metadata


def hadamard_spec() -> HadamardSpec:
    """The parsed spec, asserted non-``None`` so callers get a `HadamardSpec`."""
    spec = parse_hadamard_spec(hadamard_metadata())
    assert spec is not None
    return spec
