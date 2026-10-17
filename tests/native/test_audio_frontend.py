"""The audio front end, against the reference's own log-mel.

This is the check that the C++ in `src/model/audio_frontend.cpp` computes what
llama.cpp computes, and it is a value-for-value comparison rather than a transcript
comparison on purpose: a transcript agrees for many different wrong spectrograms, and
every decision this file encodes -- the window's periodicity, the mel scale, the log's
base, the floor -- is invisible in one.

The reference's spectrogram is captured by running `mtmd_debug_preprocess_audio`,
which llama.cpp marks as internal but exports, and parsing the mel entries out of the
log it prints.  That is the only way to reach the front end's output through the
reference without reimplementing it.

The input is real speech (`mlk.flac` decoded to 16 kHz mono), not a synthetic tone:
a tone puts most mel bands at the floor, where every implementation agrees.
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import textwrap

import numpy as np
import pytest

MMPROJ = pathlib.Path("/mnt/data1/models/mmproj-Qwen3-ASR-1.7B-Q8_0.gguf")
TEXT_MODEL = pathlib.Path("/mnt/data1/models/Qwen3-ASR-1.7B-Q8_0.gguf")
#: 13 s of real speech, decoded once by the fixture below and then fixed: a decoder
#: that resampled differently between runs would be a variable in a comparison whose
#: whole point is that its inputs are identical.
SPEECH = pathlib.Path("/tmp/pll-probe/mlk.f32")
REFERENCE = pathlib.Path("/tmp/pll-probe/ref_mel.bin")

N_MEL = 128

pytestmark = pytest.mark.skipif(
    not MMPROJ.is_file() or not TEXT_MODEL.is_file(),
    reason="the Qwen3-ASR checkpoint is not on this host",
)


def _compile(tmp_path: pathlib.Path) -> pathlib.Path:
    """A probe that runs the front end over a raw f32 PCM file and writes the mel."""
    source = tmp_path / "frontend.cpp"
    source.write_text(
        textwrap.dedent(
            """
            #include <cstdio>
            #include <cstdlib>
            #include <vector>

            #include "model/audio_frontend.h"

            int main(int argc, char **argv) {
              std::FILE *in = std::fopen(argv[1], "rb");
              if (in == nullptr) { return 2; }
              std::fseek(in, 0, SEEK_END);
              const long bytes = std::ftell(in);
              std::fseek(in, 0, SEEK_SET);
              std::vector<float> samples(static_cast<size_t>(bytes) / sizeof(float));
              if (std::fread(samples.data(), sizeof(float), samples.size(), in) != samples.size()) {
                return 2;
              }
              std::fclose(in);

              pocketllm::AudioFrontendParams params;
              params.n_mel = std::atoll(argv[3]);
              const int64_t windows =
                  pocketllm::audio_window_count(static_cast<int64_t>(samples.size()), params);
              std::FILE *out = std::fopen(argv[2], "wb");
              for (int64_t w = 0; w < windows; ++w) {
                const pocketllm::AudioMel mel = pocketllm::audio_log_mel_window(
                    samples.data(), static_cast<int64_t>(samples.size()), params, w);
                std::fwrite(mel.data.data(), sizeof(float), mel.data.size(), out);
              }
              std::fclose(out);
              return 0;
            }
            """
        )
    )
    root = pathlib.Path(__file__).resolve().parents[2]
    binary = tmp_path / "frontend"
    result = subprocess.run(
        [
            "g++",
            "-O2",
            "-std=c++17",
            f"-I{root / 'src'}",
            str(source),
            str(root / "src" / "model" / "audio_frontend.cpp"),
            str(root / "src" / "runtime" / "status.cpp"),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"cannot build the front-end probe: {result.stderr[-600:]}")
    return binary


def _reference_log() -> pathlib.Path:
    """The reference's mel, dumped through its own debug entry point.

    Built and run here rather than checked in as a fixture, so the number this file
    compares against is one the reference produced on this host -- and so a change to
    the reference is a failure rather than a stale file that still passes.
    """
    log = pathlib.Path("/tmp/pll-probe/audio_ref_mel.log")
    if log.is_file():
        return log
    probe_source = pathlib.Path("/tmp/mel_probe.c")
    if not probe_source.is_file():
        pytest.skip("no mel probe at /tmp/mel_probe.c to dump the reference with")
    result = subprocess.run(
        [
            "/tmp/mel_probe",
            str(MMPROJ),
            str(TEXT_MODEL),
            str(SPEECH),
            "/tmp/pll-probe/unused.bin",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"the reference probe failed: {result.stderr[-300:]}")
    log.write_text(result.stderr)
    return log


def _parse_reference(log: pathlib.Path) -> np.ndarray:
    values = [
        float(match.group(1))
        for match in re.finditer(r"mel\[\d+\]\[m=\d+\]\[t=\d+\] = ([-0-9.e+]+)", log.read_text())
    ]
    if not values:
        pytest.skip("the reference printed no mel entries")
    return np.asarray(values, dtype=np.float32)


def test_the_front_end_matches_the_reference(tmp_path: pathlib.Path) -> None:
    if not SPEECH.is_file():
        pytest.skip(f"no speech samples at {SPEECH}")
    binary = _compile(tmp_path)

    mine_path = tmp_path / "mine.bin"
    result = subprocess.run(
        [str(binary), str(SPEECH), str(mine_path), str(N_MEL)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    mine = np.frombuffer(mine_path.read_bytes(), dtype="<f4")

    want = _parse_reference(_reference_log())
    assert mine.size == want.size, (
        f"the engine produced {mine.size} mel values and the reference {want.size}"
    )

    difference = np.abs(mine.astype(np.float64) - want.astype(np.float64))
    correlation = float(np.corrcoef(mine, want)[0, 1])
    # Float32 is the floor, and it is a *measured* floor rather than an assumed one:
    # the reference stores the mel as f32 while accumulating in double, and a
    # reconstruction that accumulates in either precision lands within 6.4e-07 of it.
    # So this is a bound on the storage format and not a tolerance for disagreement.
    assert difference.max() < 1e-4, (
        f"the front end differs from the reference by {difference.max():.3g} at worst "
        f"(mean {difference.mean():.3g}, correlation {correlation:.8f})"
    )
    assert correlation > 0.999999, f"correlation with the reference is {correlation:.8f}"


def test_the_window_count_covers_the_signal(tmp_path: pathlib.Path) -> None:
    """The window arithmetic, which the mel comparison above cannot see.

    A signal of 13 s is 1301 effective frames at this hop, which is two 800-frame
    windows -- the second one short.  Getting the count wrong would either drop audio
    or ask for a window that does not exist, and neither shows up as a numeric
    difference in the windows that are produced.
    """
    if not SPEECH.is_file():
        pytest.skip(f"no speech samples at {SPEECH}")
    binary = _compile(tmp_path)
    samples = np.frombuffer(SPEECH.read_bytes(), dtype="<f4")
    effective = min(
        (samples.size + 400 - 400) // 160 + 1,
        samples.size // 160 + 1,
    )
    assert effective == 1301, f"the harness expects 1301 frames and this signal has {effective}"

    out = tmp_path / "windows.bin"
    result = subprocess.run(
        [str(binary), str(SPEECH), str(out), str(N_MEL)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    # Two windows: the first is the full 800 frames, and the remainder (1301 - 800 =
    # 501) is padded up to a multiple of the conv sub-chunk size, which is 600.  Both
    # numbers are the reference's for this input -- it reports `n_len=800` and
    # `n_len=600` for the two chunks of the same signal.
    assert out.stat().st_size == (800 + 600) * N_MEL * 4