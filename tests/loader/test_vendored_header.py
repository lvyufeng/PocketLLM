"""The vendored GGML header: where it is found, and that it is not quietly edited.

The loader decodes codebook formats by reading `ggml-common.h` as text. That
only works if the resolution order is what it claims and if the
byte-identity of the vendored copy is enforced, because a hand-edited table
produces plausible weights and a subtly wrong model.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pocketllm.quant import ggml_tables


def test_vendored_copy_is_the_default_source() -> None:
    path = ggml_tables.header_path()
    assert path.name == "ggml-common.h"
    assert "vendor" in path.parts


def test_vendored_copy_matches_its_pinned_hash() -> None:
    digest = hashlib.sha256(ggml_tables.header_path().read_bytes()).hexdigest()
    assert digest == ggml_tables.VENDORED_SHA256


def test_env_override_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    override = tmp_path / "ggml-common.h"
    override.write_text("GGML_TABLE_BEGIN(int8_t, kvalues_iq4nl, 16)\n0, 1, 2\nGGML_TABLE_END()\n")
    monkeypatch.setenv(ggml_tables.HEADER_ENV, str(override))
    assert ggml_tables.header_path() == override


def test_env_override_must_exist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ggml_tables.HEADER_ENV, "/nonexistent/ggml-common.h")
    with pytest.raises(RuntimeError):
        ggml_tables.header_path()


def test_a_local_edit_to_the_vendored_copy_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    tampered = tmp_path / "ggml-common.h"
    tampered.write_text("GGML_TABLE_BEGIN(int8_t, kvalues_iq4nl, 16)\n0\nGGML_TABLE_END()\n")
    # The resolver walks `Path(__file__).parent.parent / "ggml-common.h"`; point
    # the module at a fake package one level under the tampered header.
    monkeypatch.setattr(ggml_tables, "_VENDOR_RELATIVE", ("ggml-common.h",))
    monkeypatch.setattr(ggml_tables, "__file__", str(pkg / "ggml_tables.py"))
    monkeypatch.delenv(ggml_tables.HEADER_ENV, raising=False)
    with pytest.raises(RuntimeError, match="sha256"):
        ggml_tables.header_path()


def test_every_table_has_the_declared_length() -> None:
    assert ggml_tables.kvalues_iq4nl().shape == (16,)
    assert ggml_tables.kvalues_mxfp4().shape == (16,)
    assert ggml_tables.iq2xxs_grid().shape == (256,)
    assert ggml_tables.iq2xs_grid().shape == (512,)
    assert ggml_tables.iq2s_grid().shape == (1024,)
    assert ggml_tables.iq3xxs_grid().shape == (256,)
    assert ggml_tables.iq3s_grid().shape == (512,)
    assert ggml_tables.iq1s_grid().shape == (2048,)


def test_iq4_nl_codebook_is_the_known_nonlinear_set() -> None:
    # The values are deliberately unevenly spaced; pinning them is the point,
    # because a "nicer" linear table would decode every weight slightly wrong.
    assert ggml_tables.kvalues_iq4nl().tolist() == [
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    ]


def test_tables_are_read_only() -> None:
    for table in (
        ggml_tables.kvalues_iq4nl(),
        ggml_tables.iq2xs_grid(),
        ggml_tables.iq2xxs_signed_grid(),
        ggml_tables.iq2xs_signed_grid(),
        ggml_tables.iq3xxs_signed_grid(),
    ):
        assert not table.flags.writeable, "a shared table must not be mutable in place"


def test_signed_grids_have_the_expected_geometry() -> None:
    assert ggml_tables.iq2xxs_signed_grid().shape == (256, 128, 8)
    assert ggml_tables.iq2xs_signed_grid().shape == (512, 128, 8)
    assert ggml_tables.iq3xxs_signed_grid().shape == (256, 128, 8)


def test_sign_expansion_matches_the_bit_parity_rule() -> None:
    """Sign index i uses bit 7 = popcount(i) & 1, as GGML's mask does."""
    grid = ggml_tables.iq2xxs_signed_grid()
    # Codebook entry 0 is 0x0808080808080808 -- eight +8s -- so every expanded
    # value is +-8 and the sign of element j is bit j of `index | parity << 7`.
    assert set(int(v) for v in grid[0, 0]) <= {8, -8}
    for index in range(128):
        mask = index | ((index.bit_count() & 1) << 7)
        for bit in range(8):
            expected = -8 if (mask >> bit) & 1 else 8
            assert int(grid[0, index, bit]) == expected, (index, bit)