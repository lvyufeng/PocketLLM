"""Definition tests for the vLLM-convention serving benchmark client.

These pin the *definitions* - what TTFT, TPOT, ITL, E2EL, throughput and
goodput mean, and where the client's clock is read - because a timing
definition that is only checked against a live server is checked against
whatever that server happened to do.

Three layers:

* `calculate_metrics` is exercised on synthetic `RequestOutput` values, so the
  arithmetic (the `output_len <= 1` TPOT exclusion, the pooled ITL, the
  all-SLOs-met goodput rule) is compared with hand-computed numbers exactly.
* `stream_request` is driven against a stub SSE server that emits a role-only
  chunk well before any token, which is the shape both PocketLLM servers
  produce. This is the regression that keeps a client from crediting the role
  chunk as the first token.
* `arrival_delays` is checked for the two properties the scheduler relies on:
  `inf` means simultaneous, and a finite rate lands the last arrival on
  `num_requests / rate`.

No NPU, no checkpoint, no HTTP client library.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BENCH_SERVING = REPO_ROOT / "scripts" / "bench_serving.py"

ROLE_CHUNK = (
    b'data: {"id":"r","object":"chat.completion.chunk","choices":'
    b'[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
)


def _load_bench_serving():
    """Import scripts/bench_serving.py by path; scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location("bench_serving_under_test", BENCH_SERVING)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench_serving = _load_bench_serving()


def _chunk(text: str, *, finish_reason: str | None = None) -> bytes:
    body = {
        "id": "r",
        "object": "chat.completion.chunk",
        "choices": [
            {"index": 0, "delta": {"content": text} if text else {}, "finish_reason": finish_reason}
        ],
    }
    return b"data: " + json.dumps(body).encode("utf-8") + b"\n\n"


class _StubStreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for delay, payload in self.server.script:
            if delay:
                time.sleep(delay)
            self.wfile.write(payload)
            self.wfile.flush()

    def log_message(self, *args):  # keep the test output clean
        pass


class _StubServer:
    """A one-shot SSE server whose chunk timing is part of the test."""

    def __init__(self, script):
        self.script = script
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _StubStreamHandler)
        self.server.script = script
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5.0)
        return False


def _output(*, ttft: float, latency: float, tokens: int, itl=None, prompt_tokens=10, success=True):
    gaps = list(itl or [])
    return bench_serving.RequestOutput(
        success=success,
        ttft=ttft,
        ttft_first_chunk=ttft,
        itl=gaps,
        latency=latency,
        stream_end=latency,
        output_tokens=tokens,
        prompt_tokens=prompt_tokens,
        token_events=len(gaps) + 1 if tokens else 0,
    )


# ---------------------------------------------------------------------------
# The latching regression: the role chunk is not the first token
# ---------------------------------------------------------------------------


def test_ttft_tracks_the_first_token_not_the_role_chunk():
    """A role-only chunk arrives before any prefill work; it must not count.

    Both PocketLLM servers write the role delta before the backend is called,
    so a client that latches there reports a TTFT short by the whole queue and
    prefill cost. vLLM's own client has this blind spot.
    """
    script = [
        (0.0, ROLE_CHUNK),
        (0.30, _chunk("Hello")),
        (0.10, _chunk(" world")),
        (0.0, _chunk("", finish_reason="stop")),
        (0.0, b"data: [DONE]\n\n"),
    ]
    with _StubServer(script) as stub:
        honest = bench_serving.stream_request(
            stub.base_url, "/v1/chat/completions", {"stream": True}, timeout=30.0, latch="content"
        )
        parity = bench_serving.stream_request(
            stub.base_url, "/v1/chat/completions", {"stream": True}, timeout=30.0, latch="first-chunk"
        )

    assert honest.success and parity.success
    # Honest: TTFT is the token, which the script holds back by 0.30 s.
    assert honest.ttft > 0.10, honest.ttft
    assert honest.ttft_first_chunk < 0.10, honest.ttft_first_chunk
    assert honest.ttft > honest.ttft_first_chunk
    assert honest.generated_text == "Hello world"
    # Two token-bearing chunks after the first, ~0.10 s apart.
    assert len(honest.itl) == 1, honest.itl
    assert 0.03 < honest.itl[0] < 0.25, honest.itl
    assert honest.output_tokens == 2

    # Strict vLLM parity: the role chunk is accepted as the first "token", so
    # TTFT collapses to the role chunk's arrival.
    assert parity.ttft == parity.ttft_first_chunk
    assert parity.ttft < 0.10, parity.ttft
    # The role chunk, two content chunks and the empty finish chunk all count.
    assert parity.token_events == 4, parity.token_events
    assert len(parity.itl) == 3, parity.itl

    # The difference is the price of the role chunk, which is the diagnostic
    # the harness reports.
    assert honest.ttft - honest.ttft_first_chunk > 0.10


def test_latch_does_not_change_the_text_or_the_response_end():
    script = [
        (0.0, ROLE_CHUNK),
        (0.05, _chunk("a")),
        (0.05, _chunk("b")),
        (0.0, b"data: [DONE]\n\n"),
    ]
    with _StubServer(script) as stub:
        content = bench_serving.stream_request(
            stub.base_url, "/v1/chat/completions", {}, timeout=30.0, latch="content"
        )
        first = bench_serving.stream_request(
            stub.base_url, "/v1/chat/completions", {}, timeout=30.0, latch="first-chunk"
        )
    assert content.generated_text == first.generated_text == "ab"
    assert content.stream_end > 0.0 and first.stream_end > 0.0


def test_stream_without_a_token_is_a_failure_not_a_zero_ttft():
    """vLLM marks a stream with no valid chunk failed; so does this client."""
    script = [(0.0, ROLE_CHUNK), (0.0, b"data: [DONE]\n\n")]
    with _StubServer(script) as stub:
        output = bench_serving.stream_request(
            stub.base_url, "/v1/chat/completions", {}, timeout=30.0, latch="content"
        )
    assert not output.success
    assert "TTFT" in output.error


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------


def test_metrics_arithmetic_matches_hand_computed_values():
    outputs = [
        _output(ttft=0.1, latency=0.5, tokens=5, itl=[0.1, 0.1, 0.1, 0.1]),
        _output(ttft=0.2, latency=0.6, tokens=4, itl=[0.4 / 3.0] * 3),
        _output(ttft=0.3, latency=0.3, tokens=1, itl=[]),
    ]
    metrics = bench_serving.calculate_metrics(
        outputs, 2.0, percentiles=[99.0], goodput_config={"ttft": 250.0, "tpot": 150.0, "e2el": 600.0}
    )

    assert metrics["completed"] == 3 and metrics["failed"] == 0
    assert metrics["request_throughput"] == 1.5
    assert metrics["total_output_tokens"] == 10
    assert metrics["output_throughput"] == 5.0
    assert metrics["total_input_tokens"] == 30
    assert metrics["total_token_throughput"] == 20.0

    # tpot = (latency - ttft) / (output_len - 1), and the single-token request
    # is excluded from the summary but still a request.
    tpot = metrics["tpot"]
    assert abs(tpot["mean"] - (0.1 + 0.4 / 3.0) / 2.0) < 1e-12, tpot
    assert abs(tpot["median"] - (0.1 + 0.4 / 3.0) / 2.0) < 1e-12
    assert abs(tpot["std"] - 0.1 / 6.0) < 1e-12

    assert abs(metrics["ttft"]["mean"] - 0.2) < 1e-12
    assert abs(metrics["e2el"]["mean"] - (0.5 + 0.6 + 0.3) / 3.0) < 1e-12
    assert len(metrics["itl"]["percentiles"]) == 1
    assert abs(metrics["itl"]["mean"] - (0.4 + 0.4 / 3.0 * 3.0) / 7.0) < 1e-12

    # r3 fails ttft (0.3 > 0.25); r1 and r2 meet all three SLOs.
    assert metrics["request_goodput"] == 1.0
    assert metrics["output_lens"] == [5, 4, 1]


def test_metrics_alone_are_the_same_without_goodput():
    outputs = [_output(ttft=0.1, latency=0.5, tokens=5, itl=[0.1] * 4)]
    metrics = bench_serving.calculate_metrics(outputs, 1.0, percentiles=[50.0], goodput_config=None)
    assert metrics["request_goodput"] is None
    assert metrics["completed"] == 1
    assert abs(metrics["ttft"]["percentiles"][50.0] - 0.1) < 1e-12


def test_failed_requests_are_counted_but_excluded_from_the_summaries():
    outputs = [
        _output(ttft=0.1, latency=0.5, tokens=5, itl=[0.1] * 4),
        bench_serving.RequestOutput(success=False, error="HTTP 500: boom", latency=0.2),
    ]
    metrics = bench_serving.calculate_metrics(outputs, 1.0, percentiles=[99.0], goodput_config=None)
    assert metrics["completed"] == 1 and metrics["failed"] == 1
    assert metrics["output_lens"] == [5, 0]
    assert metrics["ttft"]["mean"] == 0.1
    assert metrics["request_throughput"] == 1.0


def test_goodput_requires_every_configured_slo():
    outputs = [
        _output(ttft=0.1, latency=5.0, tokens=5, itl=[1.0] * 4),
        _output(ttft=0.1, latency=0.2, tokens=5, itl=[0.02] * 4),
    ]
    only_ttft = bench_serving.calculate_metrics(outputs, 1.0, percentiles=[99.0], goodput_config={"ttft": 500.0})
    assert only_ttft["request_goodput"] == 2.0
    with_e2el = bench_serving.calculate_metrics(
        outputs, 1.0, percentiles=[99.0], goodput_config={"ttft": 500.0, "e2el": 1000.0}
    )
    assert with_e2el["request_goodput"] == 1.0


def test_unknown_goodput_key_is_rejected():
    outputs = [_output(ttft=0.1, latency=0.5, tokens=5, itl=[0.1] * 4)]
    try:
        bench_serving.calculate_metrics(outputs, 1.0, percentiles=[99.0], goodput_config={"wrong": 1.0})
    except AssertionError as exc:
        assert "wrong" in str(exc)
    else:
        raise AssertionError("an unknown goodput key must be rejected")


def test_missing_usage_leaves_input_tokens_unknown_rather_than_guessed():
    outputs = [_output(ttft=0.1, latency=0.5, tokens=5, itl=[0.1] * 4, prompt_tokens=None)]
    metrics = bench_serving.calculate_metrics(outputs, 1.0, percentiles=[99.0], goodput_config=None)
    assert metrics["total_input_tokens"] is None
    assert metrics["total_token_throughput"] is None
    assert metrics["output_throughput"] == 5.0


# ---------------------------------------------------------------------------
# Arrival schedule
# ---------------------------------------------------------------------------


def test_infinite_rate_sends_everything_at_once():
    rng = bench_serving.np.random.default_rng(0)
    delays = bench_serving.arrival_delays(8, float("inf"), 1.0, rng)
    assert delays == [0.0] * 8


def test_finite_rate_lands_the_last_arrival_on_the_configured_window():
    rng = bench_serving.np.random.default_rng(7)
    rate = 4.0
    count = 40
    delays = bench_serving.arrival_delays(count, rate, 1.0, rng)
    assert len(delays) == count
    assert delays == sorted(delays)
    assert abs(delays[-1] - count / rate) < 1e-9
    # Poisson intervals: the mean gap is 1/rate, not the rescaled total.
    gaps = [delays[i] - delays[i - 1] for i in range(1, count)]
    assert abs(float(bench_serving.np.mean(gaps)) - 1.0 / rate) < 0.02


def test_burstiness_infinity_is_a_constant_interval():
    rng = bench_serving.np.random.default_rng(0)
    delays = bench_serving.arrival_delays(10, 5.0, float("inf"), rng)
    gaps = [delays[i] - delays[i - 1] for i in range(1, 10)]
    assert all(abs(gap - 0.2) < 1e-9 for gap in gaps)


def test_burstiness_must_be_positive():
    rng = bench_serving.np.random.default_rng(0)
    try:
        bench_serving.arrival_delays(4, 2.0, 0.0, rng)
    except AssertionError:
        return
    raise AssertionError("burstiness 0 must be rejected")


# ---------------------------------------------------------------------------
# Payload shapes for the two endpoints
# ---------------------------------------------------------------------------


def test_payload_follows_the_endpoint():
    parser = bench_serving.build_parser()
    sample = bench_serving.SampleRequest(prompt="hi", expected_output_len=7, nominal_input_len=3)
    chat = benchmark_args(parser, ["--endpoint", "/v1/chat/completions"])
    completions = benchmark_args(parser, ["--endpoint", "/v1/completions"])
    chat_body = bench_serving.build_payload(chat, "m", sample, stream=True)
    comp_body = bench_serving.build_payload(completions, "m", sample, stream=True)
    assert chat_body["messages"] == [{"role": "user", "content": "hi"}]
    assert "prompt" not in chat_body
    assert comp_body["prompt"] == "hi"
    assert "messages" not in comp_body
    assert chat_body["max_tokens"] == 7 and chat_body["stream"] is True


def test_random_dataset_lengths_vary_with_the_range_ratio():
    parser = bench_serving.build_parser()
    args = benchmark_args(
        parser,
        ["--dataset-name", "random", "--random-input-len", "100", "--random-output-len", "20",
         "--random-range-ratio", "0.5", "--num-prompts", "32"],
    )
    rng = bench_serving.np.random.default_rng(3)
    requests = bench_serving.build_dataset(args, rng)
    assert len(requests) == 32
    lengths = [request.nominal_input_len for request in requests]
    assert min(lengths) >= 50 and max(lengths) <= 150
    assert len(set(lengths)) > 1
    both = benchmark_args(parser, ["--random-range-ratio", "0.0", "--num-prompts", "8", "--random-input-len", "64"])
    fixed = bench_serving.build_dataset(both, bench_serving.np.random.default_rng(3))
    assert {request.nominal_input_len for request in fixed} == {64}
    assert all(len(request.prompt) >= 64 for request in fixed)


def test_argument_parsing_helpers():
    assert bench_serving.parse_goodput(["ttft:200", "tpot:50"]) == {"ttft": 200.0, "tpot": 50.0}
    assert bench_serving.parse_goodput(None) is None
    assert bench_serving.parse_percentiles("25,50,99") == [25.0, 50.0, 99.0]
    try:
        bench_serving.parse_goodput(["ttft=200"])
    except AssertionError:
        pass
    else:
        raise AssertionError("a goodput entry without ':' must be rejected")


def test_default_percentile_metrics_match_vllm():
    parser = bench_serving.build_parser()
    args = parser.parse_args([])
    assert bench_serving.selected_percentile_metrics(args) == {"ttft", "tpot", "itl"}
    args = parser.parse_args(["--percentile-metrics", "e2el"])
    assert bench_serving.selected_percentile_metrics(args) == {"e2el"}


def benchmark_args(parser, argv):
    return parser.parse_args(argv)
