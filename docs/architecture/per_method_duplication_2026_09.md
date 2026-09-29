# Per-method duplication across the five adapters, 2026-09-29

This is the measurement #442's acceptance rests on, kept because the issue's own numbers were
estimates and the slices that followed kept needing the same question answered again: **which
same-named methods across `pocketllm/backends/*.py` are still the same body?**

The method is `scripts/method_duplication.py`: `ast` per
method, docstrings stripped, comment-only lines dropped, then `difflib.SequenceMatcher` over the
stripped statement lists for every pair of same-named methods in the six modules. What it reports is
the **best** pair per name, the line count of each side, and how many definitions the name has. A
high ratio between one pair out of five says those two agree, not that the family does.

Run it from the repository root:

```bash
python scripts/method_duplication.py
```

## Where this stood at the end of #442's slices 1–6 (master `c896925`)

| ratio | lines | method | pair |
| ---: | ---: | --- | --- |
| 1.00 | 5/5 | `generate` | v41 ~ mimo |
| 1.00 | 4/4 | `prepare` | mimo ~ xing4 |
| 0.97 | 17/16 | `_run_loop` | mimo ~ xing4 |
| 0.91 | 23/23 | `_eos_tokens` | mimo ~ xing4 |
| 0.90 | 15/16 | `capabilities` | mimo ~ xing4 |
| 0.87 | 15/15 | `_open_tokenizer` | mimo ~ xing4 |
| 0.82 | 34/34 | `_loop` | mimo ~ xing4 |
| 0.81 | 14/13 | `close` | v41 ~ mimo |
| 0.71 | 20/22 | `_start_runtime` | v41 ~ torch |
| 0.67 | 24/27 | `run_worker` | v41 ~ mimo |

Everything the six slices removed is absent from that table: `stream`, `_encode_chat`, `_result`,
`_tokenize`, `_budget`, `_decode`, `_runtime_spec` and the worker program are each one body now.

## What a ratio does and does not mean

A high ratio is where to look, not a verdict; the two rows that were opened and turned out to be
**one parameter apart** show why:

- `_run_loop` (0.97): the diff is exactly one line, `marks=marks`. `mimo._loop` takes `marks` and
  never reads it — the parameter is dead in the body and alive only in the signature. Nothing in the
  other 16 statements differs.
- `_loop` (0.82): the signature differs (`on_step` vs `marks`), the import inside differs (each
  imports its own runtime's `generate`), and the step predicate differs — `self._step_sync(...)` for
  mimo, one collective per step so every rank agrees, against a lambda folding `on_step` and `stop`
  into the local cancel flag for xing4. That last one is the whole point of the method.

`capabilities` (0.90), `_eos_tokens` (0.91) and `_open_tokenizer` (0.87) are rows where the
difference is small enough to fold behind a per-adapter hook, and all three were **left alone on
purpose**: folding a 15-line body into ~20 shared lines plus a hook is a wash in size, and it hides a
fact (which object carries the eos ids, which path resolves the tokenizer) behind a name instead of
removing it. `close` (0.81) is a fourth: mimo's sends its workers a shutdown broadcast as its first
act and xing4's drops a prefix cache as its last, and those are the two things the method is for.