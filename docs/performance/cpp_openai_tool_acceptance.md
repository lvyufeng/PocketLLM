# Native C++ OpenAI tool-calling acceptance

This document records the end-to-end acceptance test for tool calling on the native
`pocketllm_engine` OpenAI server, taken for the remaining acceptance work on #155. It
is the first record here that drives a tool conversation through **more than one
turn**, and the first that drives the server through the published `openai` and
`langchain-openai` clients rather than through bare HTTP.

Read [Benchmarking and reporting rules](../guides/benchmarking.md) before comparing
anything below; this is a correctness record, not a throughput one, and it carries no
tokens/s figures.

## Scope

The validated path is:

- Qwen3.8-27B-FP8 safetensors, `/mnt/data2/Qwen3.8-27B-FP8` (`model_type: qwen3_5`)
- native `pocketllm_engine`, full model depth (`--smoke-layers 0`)
- CUDA, TP4, physical GPUs 0-3, four RTX 2080 Ti
- OpenAI `/v1/chat/completions` and `/v1/models`, FP16 KV cache
- chat templating through `src/server/cpp_sidecar.py` with the checkpoint's own
  `tokenizer.json` (`transformers` 4.57.1, `jinja2` 3.1.6)
- tool calls supplied as OpenAI `tools` definitions and read back from the
  assistant message's `tool_calls` array
- the published clients `openai` 3.14.0 and `langchain-openai` 1.6.2
  (`langchain-core` 1.6.3, `httpx` 0.28.1, `pydantic` 2.13.5), each in a throwaway
  virtualenv so that neither the inference environment nor the system Python gets a
  client package installed into it

The following remain outside this acceptance result:

- **Streaming tool calls.** A streamed response still carries the call syntax as
  content. This is a documented limitation, not a regression, and nothing here
  contradicts it; see [PocketLLM engine API](../guides/pocketllm_api.md).
- **`tool_choice` constraints and `parallel_tool_calls`.** Both outside `"auto"` are
  400s. The model decides whether to call anything and how many calls to make.
- **Cursor and Continue.** IDE plugins cannot be driven headlessly; see
  [Not run](#not-run) below.
- **The DeepSeek-V4 sidecar templater.** It has its own DSML encoder and is not
  reached by anything below.
- **A second prompt with `--smoke-layers 4`**, which returns HTTP 500
  `prefill failed: Qwen tensor is not F32`. Unrelated to this record and tracked
  separately.

## Reproduction

The harness starts and stops all four TP ranks itself, waits for `/health`, and tears
the group down on every exit path. It imports nothing outside the standard library, so
the interpreter that runs it can be one that has the client packages while the sidecar
keeps running under the inference environment:

```bash
/tmp/venv-toolclients/bin/python scripts/verify_cpp_qwen_openai.py \
  --ckpt /mnt/data2/Qwen3.8-27B-FP8 \
  --binary cpp_engine/build-python/pocketllm_engine \
  --python /home/lvyufeng/miniconda3/envs/deepseek/bin/python \
  --devices 0,1,2,3 --require-tool-clients \
  --log-dir /tmp/tool_acceptance_logs
```

`cpp_engine/build-python` is the tree configured with NCCL
(`-DNCCL_ROOT=.../nvidia/nccl -DPOCKET_REQUIRE_NCCL=ON`). A binary built without NCCL
still runs and still answers, and is still wrong: TP all-reduce is compiled out. The
one-off virtualenv is `python3.11 -m venv /tmp/venv-toolclients` followed by
`pip install openai langchain-openai`, and nothing else.

## Results

Commit `55fb34f` on the branch's base `e2d9a17` (PR #254, the sidecar templating fix
described below), checkpoint `/mnt/data2/Qwen3.8-27B-FP8`, TP4, four RTX 2080 Ti
(driver 580.173.02), one launch, `--max-context 2048 --max-batch-size 2
--prefill-token-budget 4096 --kv-block-size 16`, `temperature 0.0`, `top_p 1.0`,
`top_k 20`.

Every check the harness already had passes at this tree — request-field refusals, stop
sequences, `n` choices, logprobs, and single-turn tool parsing — so nothing below
replaced an earlier result. The three new checks pass as well:

```text
[PASS] tool calls: 1 call(s) parsed out of the completion with arguments {"city": "Paris", "days": 3}
[PASS] multi-turn tool calls: call_7c9740e6447e4a55b29eca7f replayed with its result, answered with 196 characters of content
[PASS] openai SDK 3.14.0: models.list(), tool call call_10debec1a8f845f0956022dc, and a 196-character final answer
[PASS] langchain 1.6.2: bind_tools -> ToolMessage -> a 193-character final answer
[PASS] native Qwen OpenAI serving: tp=4 model=qwen3_5 log_dir=/tmp/tool_acceptance_logs concurrent_requests=2 tool_clients=[langchain=ran, openai SDK=ran]
```

### The two-turn exchange

Turn one asks for a forecast and offers the `get_weather` tool. The server answers
with `finish_reason: "tool_calls"` and the OpenAI shape — `arguments` as a **JSON
string**, `content` as the empty string rather than `null`:

```json
{
  "id": "req_5e26c0caf08a68c8", "object": "chat.completion",
  "model": "qwen3_5",
  "choices": [{
    "index": 0, "finish_reason": "tool_calls",
    "message": {
      "role": "assistant", "content": "",
      "tool_calls": [{
        "id": "call_9890e3f85ca44f6d8bf4ed39", "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\", \"days\": 3}"}
      }]
    }
  }],
  "usage": {"prompt_tokens": 323, "completion_tokens": 37, "total_tokens": 360}
}
```

Turn two sends that `message` object back **exactly as it arrived** — no field
dropped, none added — followed by a `role: "tool"` message keyed on that call's `id`.
`prompt_tokens` rises from 323 to 409 as the replayed message and the result are
rendered into the prompt:

```json
{
  "messages": [
    {"role": "user", "content": "What is the weather in Paris for the next 3 days? Use the get_weather tool."},
    {"role": "assistant", "content": "", "tool_calls": [
      {"id": "call_9890e3f85ca44f6d8bf4ed39", "type": "function",
       "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\", \"days\": 3}"}}]},
    {"role": "tool", "tool_call_id": "call_9890e3f85ca44f6d8bf4ed39",
     "content": "{\"city\": \"Paris\", \"days\": 3, \"forecast\": \"sunny, 21C daytime, 11C overnight\"}"}
  ],
  "tools": [ /* the same definition as turn one */ ]
}
```

The response is an ordinary completion: `finish_reason: "stop"`, no second call, and
no `<tool_call>` syntax left in the text. The model used the result.

```json
{
  "id": "req_79085cd2211bd05f", "object": "chat.completion",
  "model": "qwen3_5",
  "choices": [{
    "index": 0, "finish_reason": "stop",
    "message": {"role": "assistant",
      "content": "Here's the weather forecast for Paris over the next 3 days:\n\n☀️ **Sunny** — 21°C (70°F) during the day, dropping to 11°C (52°F) overnight.\n\nIt looks like a pleasant stretch of sunny weather ahead!"}
  }],
  "usage": {"prompt_tokens": 409, "completion_tokens": 61, "total_tokens": 470}
}
```

`usage.prompt_tokens` for turn one is visible in the rank log as
`prompt_tokens=323 max_tokens=128`, and turn two as `prompt_tokens=409`, which is how
the 86-token difference above is attributed to the replayed history rather than
assumed.

The three phrasings the harness tries are all answered; the record above is the first
of them. The harness accepts any one of them because the second turn's output belongs
to the model, not to the server, and a single oblique answer would otherwise read as a
server defect.

### Published clients

Both clients completed a full call-and-return cycle against the same server, and both
did it without `extra_body` or any other escape hatch — which is what "an OpenAI SDK
client works unmodified" means. The two differ in how they spell the generation
budget, and that difference was checked on the wire rather than assumed:

| Client | Request keys on the wire | Tool-call arguments the client hands the caller |
| --- | --- | --- |
| `openai` 3.14.0 | `max_tokens`, `messages`, `model`, `temperature`, `tools` | JSON **string** (`tool_calls[0].function.arguments`) |
| `langchain-openai` 1.6.2 | `max_completion_tokens`, `messages`, `model`, `stream`, `temperature`, `tools` (`stream` is `false`) | already-parsed **dict** (`tool_calls[0]["args"]`) |

The server accepts both spellings: `max_completion_tokens` is a first-class field on
the native request, not an alias the Python control plane synthesizes. LangChain also
sends `stream: false` explicitly, and the server answers it as a non-streaming
request.

Both clients issued two requests each — one per turn — for four
`/v1/chat/completions` calls in the run, plus the `models.list()` that the SDK check
begins with. Every one returned 200. The key sets above were read off a recording
endpoint that the two clients were pointed at directly, so they describe what each
client puts on the wire rather than what the server happened to accept.

### Controls

A client check that did not run must not be readable as a pass. The harness reports a
missing package as `[NOT RUN]` and lists the outcome of each client in the summary
line; `--require-tool-clients` promotes a not-run entry to a failure. Both directions
were run at the same commit, with the inference environment's interpreter (which has
neither client installed):

| Run | Interpreter | Flag | Observed | Exit |
| --- | --- | --- | --- | --- |
| A | deepseek env | — | `[NOT RUN] openai SDK` and `[NOT RUN] langchain`, then `tool_clients=[langchain=not run: …, openai SDK=not run: …]` | `0` |
| B | deepseek env | `--require-tool-clients` | `AssertionError: --require-tool-clients was set but these client checks did not run: langchain, openai SDK` | `1` |

Run A also re-passed the multi-turn check on its own, so the two controls differ from
the recorded run only in the client packages, not in the server path.

## The defect this found

The multi-turn check failed on its first real run, and the failure was in the server,
not in the check:

```text
AssertionError: the second tool turn was not served: HTTP 400
{"error":{"message":"TypeError: Can only get item pairs from a mapping.
  ... File "src/server/cpp_sidecar.py", line 186, in _apply
      return self._tokenizer.apply_chat_template(
  ... File "<template>", line 136, in top-level template code
  ... File "jinja2/filters.py", line 249, in do_items
      raise TypeError("Can only get item pairs from a mapping.")"}}
```

Qwen's bundled chat template walks `tool_call.function.arguments` with the `items`
filter, i.e. as an object; OpenAI specifies the same field as a JSON string. The
Python control plane has converted between the two shapes since it was written —
`template_messages()` in `pocketllm/protocol/prompt.py`, called by
`encode_chat_prompt()` — but `src/server/cpp_sidecar.py` handed its request's
`messages` straight to the checkpoint's `apply_chat_template`. Turn one therefore
worked and every turn after it failed, before the model was reached, because the
assistant message being replayed is the one the server itself had just produced.

Fixed in PR #254 (commit `e2d9a17`): `template_messages()` was made public and is now
called by `ChatTemplateTemplater.encode()`. The converted copy is what the template
sees; the request's own message list is left exactly as received, because a client
replays the object the server gave it and that value is not the server's to rewrite.
The sidecar is a Python subprocess, so no C++ rebuild was needed.

Why the earlier #155 work did not catch it: `validate_tool_calls` is single-turn. It
asks, parses the call out of the completion, and stops — which is precisely the half
of the feature that never replays the message. The acceptance criterion #155 itself
names ("a LangChain agent can call a function and return") is the other half.

## Not run

- **Cursor and Continue.** Both are graphical IDE plugins with no headless driver, so
  no script here can exercise them and no result is claimed for them. What *is*
  established is the part they depend on: an OpenAI-compatible `/v1/models` and
  `/v1/chat/completions` that a published OpenAI client drives unmodified, including a
  complete tool call and its return. Whether a given plugin's configuration dialog
  accepts a custom base URL is a fact about the plugin, not about this server.
- **Streaming tool calls, `tool_choice`, `parallel_tool_calls`, multi-call turns.**
  Listed under [Scope](#scope); each is a documented limitation or a refusal, and none
  of them was exercised here.
