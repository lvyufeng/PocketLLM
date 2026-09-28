# The command-line surface: what vLLM and SGLang do, and what PocketLLM's should be

U2b and U3 of the plan for [#447](https://github.com/lvyufeng/PocketLLM/issues/447) both change the
`serve` command line, and the two questions they raise are the same question asked twice: **which
flag spells which concept, and where does a runtime's difference live** — in the flag, or in the
value the flag takes.

Both answers are already made by the two stacks we are measured against, so this document reads
them rather than inventing a convention. It is also the reason `scripts/upstream_cli_inventory.py`
exists: the numbers below are only true of the commit they were read at, and the script reprints
them against any checkout.

| Stack | Read at | Surface |
| --- | --- | --- |
| vLLM | `004e37e` (2026-09-28) | 246 flags in 16 argument groups |
| SGLang | `f4de6ab` (2026-09-28) | 526 flags in 12 namespace classes |
| PocketLLM | `fa6b46b` (2026-09-28) | 37 `serve` flags + 34 per-runtime options = 63 distinct names |
| PocketLLM | after U2b-2 (§7) | 57 flags in 8 `--help` sections, `--help`-derived rather than counted by hand |
| PocketLLM | after U3 (§7) | 58, with `device` a host flag rather than a declaration |

Both upstream checkouts are at their tip as of the date on this document — no release tag in
between, so neither column is a description of an old version. The first PocketLLM row is the state
the design was written against; the second is where it landed, and §7 records both why the count
moved and how the tool's own counting changed with it.

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/vllm-project/vllm
cd vllm && git sparse-checkout set vllm/engine
git clone --depth 1 --filter=blob:none --sparse https://github.com/sgl-project/sglang
cd sglang && git sparse-checkout set python/sglang/srt/arg_groups

python scripts/upstream_cli_inventory.py --vllm /tmp/vllm_ref --sglang /tmp/sglang_ref \
    --ours --overlap
```

## 1. The finding that decides the design

**Neither stack has a flag spelled twice, and we have four.**

| Surface | Registrations | Distinct flags | A name used twice |
| --- | ---: | ---: | ---: |
| vLLM | 246 | 246 | 0 |
| SGLang | 526 | 526 | 0 |
| PocketLLM | 71 | 63 | **4** |

Ours are `--device` (top level and all three runtimes), `--prefill-chunk` (v41, mimo, xing4),
`--prefix-cache-bytes` (v41, mimo, xing4) and `--prefix-cache-head-tokens` (v41, mimo). This is the
state at `fa6b46b`, the commit the document was written from; U2b-1 and U2b-2 (§7) merge the four
and put the result on the command line, and U3 takes the first of them off the runtimes altogether --
`device` is a host flag now, so it is a repeat no longer. The tool reads 57 flags with no repeat at
U2b-2, and 58 at U3.

That is not a coincidence about their flags or our flags. It is what a command line is: a name on a
command line has one meaning, and a name with two meanings is a name the operator has to resolve by
knowing which runtime they are about to get. Our four exist because each runtime was free to
declare an option without asking whether another one already had the concept — and the concepts
here are the same concept:

| Ours, today | What it means in each place | The one concept |
| --- | --- | --- |
| `--device` (top level) | the process's card, or its vendor | where this rank runs |
| `device` (v41) | the base card the rank offset is applied to | |
| `device` (mimo) | the card this rank runs on | |
| `device` (xing4) | the fallback when the option is unset | |
| `prefill_chunk` | a chunk width, resolved differently per runtime | how wide one prefill forward is |
| `prefix_cache_bytes` | a host store's budget, 4g / 4g / 2g | how much prefix a rank may keep |
| `prefix_cache_head_tokens` | the head anchor, 1024 in both | |
| `expert_deal` / `deal` | v41's expert deal, default `None`; mimo's, default `"sorted"` | how the routed experts are dealt |

The last row is the useful one. `expert_deal` and `deal` are **not two options that collide** —
they are one option that two runtimes declared, with two defaults, and the defaults are the only
thing that differed. Which is the second finding.

## 2. A runtime's difference lives in the default, resolved after the parse

SGLang is the clear statement of this and it is worth reading in full, because it faces exactly our
problem: DeepSeek-V4's sparse attention, on CUDA and on NPU, at 256 page size and at 128.

`arg_groups/model_overrides/deepseek_v4.py`:

```python
@_register_for("DeepseekV4ForCausalLM")
def _deepseek_v4_overrides(server_args, hf_config) -> dict:
    overrides = {"attention_backend": "dsv4"}
    ...
    page_size = 256
    if cfg.device == "npu":
        page_size = 128
        overrides["prefill_attention_backend"] = "dsv4"
        overrides["decode_attention_backend"] = "dsv4"
    overrides["page_size"] = page_size
    ...
    if cfg.moe_runner_backend == "auto":
        if <the checkpoint is NVFP4>:
            overrides["moe_runner_backend"] = "flashinfer_trtllm_routed"
```

Every key it sets is a **global flag's field** — `--attention-backend`, `--page-size`,
`--moe-runner-backend` — and every one of those flags exists regardless of the model. What the
model changes is the value. The machinery around it:

- `Arg(resolvable=True)` marks a field as one a resolution pass may write (`resolvable_fields` is
  the whitelist; `Arg` in `arg_groups/arg_utils.py` carries it beside `help`, `choices`, `aliases`,
  `cli_name`, `no_cli` and `fallback`).
- `declare_resolution` records a value; `resolving_view` / `resolved_view` read through the
  declarations. Both views are **read-only** — `__setattr__` raises — so a pass cannot quietly
  overwrite a global field, it can only shadow it for a reader.
- Precedence is last-writer-wins, and `handle_model_specific_adjustments` dispatches on
  `hf_config.architectures[0]`.

The refusals live in the same place, which is the other half of the pattern: `qwen4_exp.py` raises
`ValueError` when `--pp-size > 1` is combined with MORI disaggregation, and vLLM's
`validate_flashinfer_moe_ep_model` raises when a flashinfer `moe_ep` backend is named for a model
that has no such path. A combination that cannot be served is refused **by name**, not silently
retuned.

**We already do this.** `_Options.from_args` resolves a runtime's levers once at construction, and
the two places it cannot produce a constant are already resolution rather than a flag: Xing4's
`prefill_chunk` is derived from the card's free memory and the context by `_chunk_for`, and
`CppBackend`'s per-request sampling refusal reads the engine's own capability object. So the
`--prefill-chunk` collision does not need a name per runtime, it needs `unset` to keep meaning
*resolve it*.

## 3. Where upstream does put a family prefix

I expected to find that neither stack prefixes a flag by model, and that is wrong — SGLang does, in
four places, and reading them is what makes the rule:

| SGLang | What it is | Why it is not a shared flag |
| --- | --- | --- |
| `--dsv4-attn-backend`, `--dsv4-prefill-backend` | DeepSeek-V4's sparse MLA backend choice | `--attention-backend` still exists beside it and means the general choice |
| `--dsa-decode-backend`, `--dsa-prefill-backend`, `--dsa-topk-backend`, `--dsa-paged-mqa-logits-backend` | the DSA indexer's kernels | there is no indexer outside these models |
| `--kt-weight-path`, `--kt-method`, `--kt-num-gpu-experts`, `--kt-cpuinfer`, `--kt-threadpool-count` | KTransformers, a *backend* | the knobs have no meaning in another backend |
| `--cuda-graph-bs-prefill`, `--cuda-graph-max-bs-decode`, … (11) | CUDA graph geometry | nothing else has a captured graph |

So the rule is **not** "never prefix a family". It is:

> A flag is unprefixed when the concept is shared and prefixed when the concept belongs to one
> family — and the family is named after what owns the concept, which is sometimes a model
> (`--dsv4-*`) and sometimes a backend (`--kt-*`).

Two consequences for us. First, `--expert-*` is the right shape for the arena knobs: vLLM has
`--expert-placement-strategy`, `--enable-expert-parallel`, `--ep-size`; SGLang has `--ep-size`,
`--ep-dispatch-algorithm`, `--init-expert-location`. The concept is shared across runtimes (v41's
arena and MiMo's arena are different registrations of it), so it is a family prefix, not a runtime
prefix. Second, almost nothing in our surface is a *single-runtime* concept — see §5 — so almost
nothing needs a runtime name, and the runtime belongs in the **group**.

## 4. `--help` is organized by group, never by name

vLLM registers every flag inside `parser.add_argument_group(title="CacheConfig")` — 16 groups,
51 flags in `ParallelConfig` alone. SGLang's 12 namespaces (`model`, `parallel`, `memory`,
`schedule`, `serving`, `spec`, `disagg`, `lora`, `mm`, `device`, `observability`, `exec_`) come
from the file a field is declared in, and `_field_to_cli_name` never consults them: the flag is
still `--max-running-requests`, flat.

Neither stack has a vendor group or a model group. The grouping is by subsystem, and it is what a
reader browses when they do not know a flag's name. Our `BackendOption` needs a `group` for the
same reason — and the group is where "*this is a v41 lever*" is said.

The dotted form is a separate mechanism and worth keeping in mind as the shape of a long tail:
vLLM's `--compilation-config.mode` / `-cc.mode`, and the JSON-or-dotted typing on `--logging-config`,
`--kv-transfer-config`, `--speculative-config` and a dozen more. That is what our `--backend-option`
already is.

## 5. The mapping

Every flag we have, against its counterpart. `—` means no upstream flag does the job.

### Shared concepts — one flag each

| Ours | vLLM | SGLang | Verdict |
| --- | --- | --- | --- |
| `--model` | `--model` | `--model-path`, alias `--model` | keep |
| `--max-model-len` | `--max-model-len` | `--context-length` | keep |
| `--dtype` | `--dtype` | `--dtype` | keep |
| `--kv-cache-dtype` | `--kv-cache-dtype` | `--kv-cache-dtype` | keep |
| `--tensor-parallel-size` | `--tensor-parallel-size`, `-tp` | `--tp-size`, `--tensor-parallel-size` | keep; add `--tp-size` alias |
| `--host`, `--port` | `--host`… (uvicorn) | `--host`, `--port` | keep |
| `--served-model-name` | `--served-model-name` | `--served-model-name` | keep |
| `--tokenizer-path` | `--tokenizer` | `--tokenizer-path` | keep |
| `--model-format` | `--load-format` | `--load-format` | keep the name, note the counterpart |
| `--enable-prefix-caching` | `--enable-prefix-caching` | `--disable-radix-cache` (inverted) | keep; ours matches vLLM |
| `--max-batch-size` | `--max-num-seqs` | `--max-running-requests` | keep |
| `--speculative-method`, `--speculative-tokens` | `--spec-method`, `--spec-tokens` | `--speculative-algorithm`, `--speculative-num-steps` | keep; add `--spec-tokens` alias |
| `--device` | `--device` (`auto\|cuda\|cpu\|tpu\|xpu`, deprecated) + `--device-ids` | `--device` (`cuda\|xpu\|hpu\|npu\|cpu\|musa`) + `--base-gpu-id`/`--gpu-id-step` | **split** — see below |
| `--pd-mode` | `--kv-transfer-config` (a dict) | `--disaggregation-mode` | keep |
| — | `--enforce-eager`, `--disable-cuda-graph` | `--disable-cuda-graph` | we have no such flag; our graph capture is a backend option |

### The four collisions — one flag each after the merge

| Today | After | How the difference is carried |
| --- | --- | --- |
| `--prefill-chunk-tokens`, `prefill_chunk` ×3 | `--prefill-chunk-tokens` | the flag reaches all three; `unset` = the runtime's own fallback: v41 → the loader's width, mimo → 2048, xing4 → `_chunk_for(context, heads, free_bytes)` |
| `--prefix-cache-bytes` ×3 | `--prefix-cache-bytes` | `unset` = the runtime's own constant (4g / 4g / 2g) |
| `--prefix-cache-head-tokens` ×2 | `--prefix-cache-head-tokens` | 1024 in both; keep one declaration referenced by both runtimes |
| `--expert-deal` (v41, default `None`), `deal` (mimo, default `"sorted"`) | `--expert-deal`, choices `sorted\|id` | `unset` = resolve: v41 → the loader's own; mimo → `POCKETLLM_MIMO_EXPERT_DEAL`, then `"sorted"` |
| `--device` (top level) + `device` ×3 | `--device` + `--device-ids` | see below |

### `--device` splits into two flags

Today one name does two jobs, and the two jobs disagree: the top level refuses `--device` outright
under automatic TP supervision, while v41 reads its own `device` twice over — verbatim as the card
its dense tree sits on, and as the *base* card the rank offset is applied to when it is deciding
where the experts go — mimo ignores `getattr(args, "device")` entirely, and xing4 falls back to it.

Upstream separates them, and both do it the same way:

| | vendor / platform | which cards |
| --- | --- | --- |
| vLLM | `--device auto\|cuda\|cpu\|tpu\|xpu` (default `auto`, deprecated in favour of detection) | `--device-ids 2,3` |
| SGLang | `--device cuda\|xpu\|hpu\|npu\|cpu\|musa`, `None` = auto-detect | `--base-gpu-id`, `--gpu-id-step` |

For us: **`--device auto|cuda|ascend`** (the answer comes from `pocketllm_cpp.backend` or platform
detection, and an explicit value this build cannot serve is a hard error — vLLM's
`validate_flashinfer_moe_ep_model` is the precedent for refusing rather than retuning), and
**`--device-ids 2,3`** for the cards, comma-separated and in rank order, defaulting to the rank.
`CppBackend._native_rank_device` already derives the rank's card; this is that rule given a name.

It also removes the refusal rather than restating it: `--device` (a vendor) has no reason to
conflict with automatic supervision, and `--device-ids 2,3` is well defined under it — rank *r*
takes `device_ids[r]`.

As landed (§7), three details of the above are not what the sketch said, and each is recorded where
it is decided: `cpu` is in the platform set, because a host-only run is one this repository makes
and upstream keeps a `cpu` for the same reason; *where* the flag lives is the host rather than a
declaration, because the card list means one thing on all four adapters including the one that
cannot declare; and the ids are indices into the set the process can see rather than physical,
which is what keeps the `CUDA_VISIBLE_DEVICES` form working.

### Ours only, and why that is not a defect

| Ours | Closest upstream | Why there is no counterpart |
| --- | --- | --- |
| `--backend` | `--attention-backend`, `--moe-runner-backend` (`auto` + choices) | upstream has one engine per process; the runtime is a build/installation fact. Ours is a choice, so it stays — as an explicit override that wins over `auto`'s answer, which is what an upstream `auto`-plus-choices flag is |
| `--enable-batching` | — | upstream has no serialized path to fall back to; their analogue is a width of 1 |
| `--engine-kind` | `--model-impl` | our native registry's choice between the persistent and Qwen engines |
| `--tensor-parallel-rank`, `-supervisor`, `-startup-timeout`, `-shutdown-timeout`, `-master-addr`, `-master-port`, `-rendezvous-dir` | `--nnodes`, `--node-rank`, `--master-addr`, `--master-port`, `--distributed-timeout-seconds`, `--dist-init-addr`, `--dist-timeout` | upstream delegates rank launching to torchrun and never supervises in-process. Ours supervises, so the family is ours; the address/timeout trio should take upstream's names |
| `--routed-experts-device` | `--cpu-offload-gb`, `--offload-backend` | theirs is a byte budget, ours is a placement |
| `--attention-window`, `--attention-sink-tokens` | `--disable-sliding-window`, `--swa-full-tokens-ratio` | theirs is a switch and a fraction; ours is the window and the sink in tokens |
| `--backend-option` | `--hf-overrides`, `--compilation-config.*`, `--kv-transfer-config` | the same idea, and the one that stays as the long tail and the alias (§6) |
| `--supervised-child` | hidden flags, e.g. SGLang's `---x-explicitly-set` | internal, and already `argparse.SUPPRESS` on our side |

**The per-runtime options that upstream has no flag for at all** are the widest gap, and it is the
Project #7 decision showing through: `expert_device`, `expert_world`, `expert_cache`,
`expert_hot_rows`, `expert_pool_rows`, `expert_buffers`, `expert_batched`, `resident_engram`,
`resident_experts`, `decode_graphs`, `cancel_collective`, `threads`, `skip_special_tokens`,
`progress` (v41); `chunk_rows`, `pin`, `resident_rows`, `slots` (mimo); `gguf`, `tokenizer`,
`use_kernel` (xing4). Upstream's placement knobs are global — `--cpu-offload-gb` is one number for
the whole engine — because their engine is one implementation. Ours are per-runtime because ours
are separate implementations, which is the point of the refactor. They get **flat names in a
per-subsystem group**, with the family prefix where they have siblings (`--expert-*`, `--resident-*`),
and `--help` says which runtime reads them.

## 6. The rules

1. **A flag names one concept.** The four collisions are merged, and a runtime that reads a merged
   flag is one more reader of it.
2. **A difference that is a value is a resolved default, not a flag.** `unset` means *resolve*; a
   resolver keyed on the runtime (and, later, on the checkpoint's architecture the way SGLang keys
   on `hf_config.architectures[0]`) supplies it. Precedent: SGLang's `deepseek_v4.py` sets
   `page_size` 256/128 by device; ours sets Xing4's chunk width from the card.
3. **A flag keeps an unprefixed name when the concept is shared, and takes a family prefix when it
   belongs to one family** — `--expert-*`, `--resident-*`, and `--dsv4-*`/`--kt-*` are the shape of
   that. A *runtime* prefix is the last resort, not the default: `--help`'s group is where "v41
   reads this" belongs.
4. **An enumerated flag whose value is normally computed carries `auto`**, and an explicit value
   that cannot be served is a refusal naming the flag — never a silent retune.
5. **`--help` is grouped by subsystem.** `BackendOption` gains a `group`, and the generated help
   inverts today's `--backend-option KEY=VALUE` listing into something a reader can browse.
6. **`--backend-option` stays**, as the alias for a generated flag and as the escape hatch for a
   key with no flag — the same role `--hf-overrides` and `--compilation-config.*` play upstream,
   and the reason `scripts/` does not have to be migrated in one step.

What we deliberately do **not** copy: their flag *count*. vLLM's 246 and SGLang's 526 are mostly
multi-vendor and multi-modality surface (Mamba, LoRA, multimodal preprocessing, four accelerator
backends, PD disaggregation) that we do not have. Our surface should stay near seventy, and the
test of a new flag is rule 1, not a comparison of counts.

One SGLang mechanism we also do not need: `---x-explicitly-set` (a triple-dash mark per field,
tested by resolution passes to answer "did the operator name this?"). Our resolved values are
computed once in `_Options.from_args` from a `None` that means unset, so the question is already
answered by the value. SGLang needs the mark because its passes write into a shared record that has
already been populated by the parse.

## 7. What this changes in the plan

The three steps are unchanged in order; this document fixes their contents.

**U2b-1 — structure, one fix.** `BackendOption` gains `group`, `resolution` and `readers`; the five
rows of §5's collision table become one declaration each in `pocketllm/backends/shared_options.py`,
referenced by every runtime that reads it and differing only in what that runtime answers;
`decode_options` takes the resolved mapping as well as the raw `backend_options` dict. `--help`
still lists `--backend-option` and the flag set is identical.

One behaviour does change, and it is a fix the structure exposed rather than a decision: MiMo's
adapter read no `--prefill-chunk-tokens` at all, so a MiMo launch that named it prefillled at the
runtime's own 2048 regardless — which is the silent no-op the merge exists to end. The flag now
reaches all three runtimes, and each runtime's *unset* answer is the one it always had. The aliases
widen by one name for the same reason: `--expert-deal`'s reader list makes v41 accept `deal=` too,
which is what one declaration read by two runtimes means.

`tests/test_declared_options.py` holds the tree to this: the shared list is exactly the set of names
more than one runtime declares, `readers` matches the tree, and every reader agrees with the shared
shape on everything except the three fields a runtime answers for itself.

**U2b-2 — land the flags.** As landed:

* Every declaration generates one flag, `pocketllm/backends/cli_surface.py`, registered under its
  `group` with `add_argument_group`. **One namespace holds all three runtimes' flags** — vLLM's
  parser is built from one config struct and SGLang's from one flat field list, and a namespace per
  runtime would be a namespace the operator has to know the name of before `--backend auto` has
  answered. MiMo's `--chunk-rows` and V4.1's `--expert-pool-rows` sit in one *Expert arena* section.
* **A flag the selected runtime does not read is refused, by name, with the runtimes that do.**
  `factory.select_backend` is where it belongs: it is the first place the runtime is known. Same
  sentence and same reason as `--backend-option`'s own refusal — a tuning option that silently does
  nothing is how a run ends up measured on the wrong lever.
* **`--backend-option` is the more specific spelling and wins**, which is `decode_options`'s
  existing tiers: the key layer is `backend_options`, the flags are `resolved_options`, and the
  order between them is stated once. A key with no flag keeps working, which is why U2b-2 needs no
  script migration.
* **Two declarations have no generated flag.** `prefill_chunk` is spelled `--prefill-chunk-tokens`
  by the host, the native engine having read that name since before the declarations existed;
  `device` is U3's, because today's `--device` means the vendor on one path and the card on another
  and a generated one would be a second meaning for a name that has one.
* **A flag nobody named is absent rather than sentinel-valued.** Every generated action is
  registered with `argparse.SUPPRESS` and writes into one mapping, so the mapping's keys *are* the
  options the launch named — which is what the refusal reads. That is also why §6's note about
  SGLang's `---x-explicitly-set` marks stands: our parse has no second pass to tell itself apart
  from, so "did the operator name this?" is a question about presence.
* **`--help` says who reads each flag** (`v41 only`, `v41 and mimo`) and what each of them answers
  when nobody names it (`when unset: 4g on v41 and mimo; 2g on xing4`). It has to: the section is
  the subsystem, not the runtime, and §6.3 rules out a runtime prefix for a shared concept.
* The flag set is otherwise identical, no script changes, and the four merged names keep one of
  their current spellings.

Re-run at that commit, the tool reads **57 registrations, 57 distinct names** for us against
246/246 and 526/526. Two things moved it from §1's 71/63, and both are worth stating: the merge
(the same four names are now declared once) and the counting (the tool now counts one spelling per
flag, because a `BooleanOptionalAction`'s `--no-` half is not a literal in anybody's source —
upstream's are equally invisible to the reading this does for them).

`tests/test_cli_declared_options.py` is where the generation is checkable, against the parser's own
actions rather than against a second list: every generated flag is its declaration's name, no
generated flag collides with one the host declares, a section exists for every group a declaration
names, the mapping holds exactly what the launch named, and the refusal names the flag and its
reader.

**U3 — `--device` splits.** As landed:

* **`--device auto|cuda|ascend|cpu` is the platform, `--device-ids 2,3` is the cards**, and rank *r*
  takes the r-th entry. `cpu` is not in the sketch above and is in the set, because upstream keeps
  one in a list of accelerators for the same reason we do: a value the build cannot serve is refused
  rather than retuned, and a host-only run is one this repository really makes. The card list is
  indices into the set the process can see, which is what keeps `CUDA_VISIBLE_DEVICES=$rank` +
  `--device-ids 0` meaning what the old pair meant.
* **Both are host flags, and the `device` declaration is gone rather than renamed.** This is the one
  place the plan above was wrong, and the reason is the split's own result: after it, the card list
  means the same thing on all four runtime adapters *including* `cpp`, which has no `OPTIONS` list
  and so cannot declare anything. A flag every runtime reads is a fact about the launch, like
  `--tensor-parallel-size`, and it lives beside it in `EngineArgs`. The refusal in U2b-2 is what
  settles it: a flag exempt from "the selected runtime does not read this" is not in that system.
  The reading moved to `runtime_engine.card_for_rank`, so the four adapters ask one question once.
* **The 39-file migration above is wrong, and the tool that counted it was counting names.**
  Reviewed against `scripts/` at `f739286`: of the 40 files mentioning `--device`, `CUDA_VISIBLE_DEVICES`
  or `ASCEND_RT_VISIBLE_DEVICES`, all but two are passing `--device` to something *else* — the native
  binary's own smoke/bench front end (`$BIN … --tp-rank $rank --device 0`), which U3 does not touch
  and U1e does not delete (it deletes that binary's *serving* front end), or a bench script's own
  argparse parser taking a torch device string (`--device cuda:2` to `torch.device`). **No script in
  the repository passes `--device` to `python -m pocketllm serve`**, which is the only command line
  U3 changes, so the in-repo migration is zero files. What is left is the honest half of the claim:
  the change is breaking for anybody's *own* launcher, and it gets a note.
* **The old spelling is refused by name, with the new flag in the message**, in the parser and again
  in `EngineArgs.__post_init__` — one function, `api.device_hint`, because two places refuse the same
  well-formed value and neither can rely on the other having run. `--backend-option device=…` is
  refused too, as an undeclared key, which is what removing the declaration means.
* **The refusal under automatic supervision is gone with it**, rather than restated: a platform has
  nothing to conflict with supervision, and a card list is well defined under it.
* `cli_surface.NO_FLAG` is the one line that changes, and it is now empty.

## Verification

`scripts/upstream_cli_inventory.py` produced §1's table and §3's families, from the checkouts named
at the top. It reads both upstream projects as text — no import, no dependency — so it re-runs
against a future commit without installing anything, which is the property that makes the numbers
in this document checkable rather than asserted. Since U2b-2 its PocketLLM half reads the parser the
CLI builds rather than the declarations, so the flags it reports are the ones an operator can type,
and it lists the repeated names, what stands behind each, and the declarations that have no flag of
their own -- one before U3 (`prefill_chunk`), none after it.
