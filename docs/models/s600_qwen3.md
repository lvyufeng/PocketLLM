# Qwen3 on the RDK S600 (Horizon Nash BPU)

This page is the durable record of what this tree can actually run on the **RDK S600** board and how
to run it: the command, the two environment variables it cannot start without, the measured
performance ladder, the memory ceiling that stops it at 1.7B, and the determinism contract. Every
number and every failure below was measured on the board on 2026-10-09 — whether it transfers to
another S600 is a claim to re-check in place, and the SDK version is part of the result.

Two facts frame the rest, and both are easy to get wrong:

- The S600 is **not the C engine's target.** There is no CUDA, no compiled `libpocketllm.so` for
  aarch64 here, and `--device horizon` does not select a [Python backend](../architecture/devices.md)
  either — those are stubs. It selects the **`xlm` delegate**, an `EngineBackend` adapter
  (`python/pocketllm/server/xlm_backend.py`) over D-Robotics' `libxlm.so`, which runs a
  prebuilt `.hbm` graph on the BPU. See [Serving](../architecture/serving.md) for the adapter
  contract and [#571](https://github.com/lvyufeng/PocketLLM/pull/571)/[#572](https://github.com/lvyufeng/PocketLLM/pull/572)
  for how `run` and `serve` reach it.
- **The delegate is text-in / text-out.** It tokenizes, applies its own chat template, decodes on the
  BPU, and hands text back. There is no logits surface and no token-id surface (the SDK's
  `XLM_INPUT_TOKEN` path is "not support yet"), which is why the sampling rules in
  [Sampling and caps](#sampling-and-caps-the-cli-enforces) are what they are: there is nothing on
  this path to sample from.

## The board and what it runs

RDK S600 — Horizon **Nash** BPU, aarch64, **4 BPU cores**, Linux 6.1.158-rt, Python 3.12.3. The
D-Robotics LLM SDK (`D-Robotics_LLM_S600_1.0.2_SDK`) ships the runtime (`oellm_runtime/`), a
compiler tree (`oellm_build/`), tokenizer + `generation_config.json` under
`oellm_runtime/configs/Qwen3_config/`, and per-size `.hbm` graphs under `oellm_runtime/model/Qwen3_*/`.

Two of the four shipped Qwen3 sizes load and generate:

| Size | `.hbm` | Precision | File size | Loads? |
|---|---|---|---|---|
| Qwen3-0.6B | `Qwen3-0.6B_language_chunk_512_cache_4096_w8_nash-p_corenum_4_4.hbm` | w8 | 1.02 GiB | **yes** |
| Qwen3-1.7B | `Qwen3-1.7B_language_chunk_512_cache_4096_w4_nash-p_corenum_4_4.hbm` | w4 | 1.70 GiB | **yes** |
| Qwen3-4B | `Qwen3-4B_language_chunk_512_cache_4096_w4_nash-p_corenum_4_4.hbm` | w4 | 3.10 GiB | **no** — see [the ceiling](#the-4b-8b-ceiling) |
| Qwen3-8B | `Qwen3-8B_language_chunk_512_cache_4096_w4_nash-p_corenum_4_4.hbm` | w4 | 5.31 GiB | **no** |

The naming is the compiler's: `chunk_512` is the prefill chunk, `cache_4096` the max context,
`corenum_4_4` the BPU-core assignment. Every large model in the SDK ships as `corenum_4_4`; **there
is no smaller-corenum 4B/8B variant to fall back to.**

## Running it: the exact contract

Two environment variables must be set **before `pocketllm run` starts**, and the CLI requires them
rather than setting them:

```bash
SDK=~/llm_sdk/D-Robotics_LLM_S600_1.0.2_SDK

export LD_LIBRARY_PATH=$SDK/oellm_runtime/lib:$LD_LIBRARY_PATH   # read by dlopen at process start
export HB_DNN_USER_DEFINED_L2M_SIZES=6:6:6:6          # the L2m split the .hbm was compiled for

PYTHONPATH=python python -m pocketllm run \
    --model $SDK/oellm_runtime/examples/llm_demo/qwen3_1.7b_config.json \
    --device horizon \
    --prompt "The capital of France is"
```

`--model` accepts either an `.hbm` path or the SDK's demo-style JSON config (the `*_config.json`
above); the config names the `.hbm`, the tokenizer directory, `bpu_core` and `model_type`
(Qwen3 = `9`). `--tokenizer-path` / `--config-path` override what the JSON says. `run` and `serve`
resolve `--model` through the **same** function (`_resolve_model`), so both accept the same spelling.

**Both variables are required, and the failure is only loud if we make it so.**
`LD_LIBRARY_PATH` must point at **`$SDK/oellm_runtime/lib`** — the SDK's libraries (both
`libxlm.so` and `libopencv_world.so.409`) live under `oellm_runtime/`, and there is **no** `$SDK/lib`
directory. A path guessed as `$SDK/lib` is not an error on its own; it fails later as the confusing
missing-`libopencv_world.so.409` this paragraph used to blame, which is why the correct directory is
named here and in the block above. `LD_LIBRARY_PATH` is read by `dlopen` **before the process
starts** — exporting it from inside Python is a no-op, so "set it for the user" cannot fix it.
`HB_DNN_USER_DEFINED_L2M_SIZES=6:6:6:6` is the L2m split the graph was compiled with, and the SDK's
own `run_llm.sh` sets exactly `6:6:6:6` for every model. `_require_delegate_env()` fails with a
message naming the missing variable(s) before touching a 1 GiB `.hbm`; it does not silently default.

### Reading it in `pocketllm devices`

`pocketllm devices` lists the **Python** backends and their stubs, so on the board it prints

```text
horizon    horizon   missing the Horizon OpenExplorer runtime (libhbrt4.so) and a BPU device  aot_compile       14 ops
```

and that line says nothing about whether the delegate runs — the delegate is reached through `xlm`,
not through the ABI backend of the same name. The `horizon` kind is what `--device horizon` selects;
`aot_compile` there is the *unimplemented* AOT path, not the `.hbm` delegate.

## Measured ladder

Fixed prompt **"The capital of France is"**, greedy `generation_config.json`, through
`pocketllm run --device horizon`. Throughput and load time from the delegate's `last_performance()`
(`xlm_model_performance_t`); three runs per size.

| Size | Loads? | Load | Prefill | Decode | TTFT |
|---|---|---|---|---|---|
| Qwen3-0.6B (w8) | yes | 5.25–5.30 s | 6169 t/s | **87.0 t/s** (86.27 / 87.17 / 87.39) | not available |
| Qwen3-1.7B (w4) | yes | 5.62–5.69 s | 5172 t/s | **69.4 t/s** (69.35 / 69.55 / 69.31) | not available |
| Qwen3-4B (w4) | no | — | — | — | — |
| Qwen3-8B (w4) | no | — | — | — | — |

**TTFT is not available on this SDK build.** The `ttft`, `tpot` and `end_to_end_cost` fields of
`xlm_model_performance_t` come back `0.0`, so a time-to-first-token is not something this page can
quote from the runtime; wall-clock load time above is quoted instead. Decode at 87 t/s (0.6B) and
69 t/s (1.7B) is the load-bearing number: it is what a caller feels, and the spread across three runs
is under 1.4% at 0.6B and under 0.4% at 1.7B, so it is stable rather than a lucky run.

## The 4B / 8B ceiling

4B and 8B **refuse to load**, and the refusal is the board's, not this tree's:

```text
Cannot malloc bpu memory with length 3326941192 bytes   # 4B
Cannot malloc bpu memory with length 5703561320 bytes   # 8B
  -> HBRT4_STATUS_RESOURCE_EXHAUSTED
  -> ion_alloc ret=-12 (ENOMEM)
  -> hbDNNInitializeFromFiles error code -400001
```

`hbDNNInitializeFromFiles` (**HBRT 4.7.5**, `hbrt4_mem::unified`) reads the whole `.hbm` into **one
contiguous ION buffer** and hands it to the BPU. Measured on the board with `hb_mem`'s own
`/sys/kernel/debug/ion/heaps/` accounting, holding 1.7B open: the `.hbm` lands as a **single
1,829,109,760-byte (`label hbm`) allocation in the `ion_carveout` pool**, which is **2.00 GiB**. The
4B graph needs a **3.10 GiB** contiguous allocation and the 8B **5.31 GiB**, both larger than the 2 GiB
pool — which is the whole of the refusal.

> **Correction to an earlier reading of this page.** 384 MiB was attributed here to `bpu_region` and
> called the pool that overflows. That was wrong: the `.hbm` is not `bpu_region`, and enlarging it
> would not have helped — see [what the pools are](#what-the-pools-are) below.

**This is not our Python.** Three independent checks:

- The SDK's **own** demo binary (`oellm_runtime/examples/llm_demo/llm`) on the same 4B/8B configs
  fails identically, while it initializes 0.6B/1.7B.
- `hrt_model_exec model_info` — the SDK's own tool — reports the same `HBRT4_STATUS_RESOURCE_EXHAUSTED`.
- The 4B/8B `.hbm` md5s match the SDK's published `md5sum.txt`, so the artifacts are not corrupt.

`cli.py` / `xlm.py` surface the delegate's refusal as a clean message and `rc 1`, with no traceback.

**No board or SDK knob moves it.** Measured, all through the SDK's own binary so none of our code is
in the path:

| Knob tried | Values | Effect on 4B |
|---|---|---|
| `HB_DNN_USER_DEFINED_L2M_SIZES` | `6:6:6:6`, `0:0:0:0`, `2:2:2:2`, `1:1:1:1`, `12:12:12:12` | none — identical `ion_alloc -12` |
| `bpu_core` (corenum) | `[0]`, `[0,1]`, `[0,1,2]`, `[0,1,2,3]` | none — identical |
| HBRT memory-mode env var | — | no such variable exists (`strings` over `libxlm.so` / `libhbrt4` / `libhbipm` show only `HBTL_*` diagnostics) |

Corenum cannot help because the refusal happens at `hbDNNInitializeFromFiles`, **before** any core is
assigned. The L2m split cannot help because the failure is one contiguous allocation's size, not its
L2m partition — and the SDK's docs set `6:6:6:6` for every model regardless of size, so it is not a
size-dependent lever.

### What the pools are

The board's DRAM carve-outs are fixed device-tree `reserved-memory` nodes (**not** kernel cmdline —
`/proc/cmdline` has no memory argument at all). Read from the live DT and confirmed against the boot
blob `/boot/hobot/rdk-s600-mcb-v1p0.dtb` (source `/boot/rdk-s600-mcb-v1p0.dts`), the model draws on
three 2 GiB ION heaps and ignores the one node this page used to blame:

| DT node | `compatible` | Address | Size | Role (measured) |
|---|---|---|---|---|
| `bpu_region@4300000000` | *(none)* | `0x408c000000` | **384 MiB** | `no-map`; **not an ION heap**, and a 1.7B load allocates nothing in it — **not involved in the .hbm load** |
| `ion_reserved@40C0000000` | `ion-pool` | `0x40c0000000` | 2.00 GiB | HBRT workspace — 1.7B uses **1.48 GiB** (activations + KV) |
| `ion_carveout@4140000000` | `ion-carveout` | `0x4140000000` | **2.00 GiB** | **the `.hbm` weights** — 1.7B = one 1.70 GiB `hbm` buffer; **the binding limit** |
| `ion_uncache@400000000` | `ion-uncache` | `0x4200000000` | 2.00 GiB | per-core BPU scratch — 1.7B uses ~0.17 GiB |

A full Qwen3 load therefore consumes **≈ 1.70 GiB (carveout) + 1.48 GiB (pool) ≈ 3.2 GiB**; the 2 GiB
`ion_carveout` is what a 3.10 GiB 4B `.hbm` cannot fit into. `bpu_region` is a legacy `no-map` reserve
with no ION personality and no consumer we could observe.

### Is it resizable? In principle yes; not by us, safely

The pool sizes are read from the device tree, not hard-coded, and the boot DTB is a normal writable
file (`/boot` is `boot_a`, the ext4 `boot_cur` slot, mounted `rw`). A candidate has now been **built
and validated offline** — see [Preparing the resize](#preparing-the-resize-offline) below — but it was
**not** applied and the board was **not** rebooted, for the reasons there. In outline: a 4 GiB
`ion_carveout` would let the shipped 3.10 GiB 4B load **with no recompile**, if it boots.

- **No kexec** (`CONFIG_KEXEC` unset; no `kexec` binary), so there is **no in-place, revert-on-failure
  test** — any change is only exercised by a full reboot.
- `/proc/cmdline` shows **`hobotboot.secureboot=1`**. The live DTB is selected through the A/B slot
  machinery (see below), which is now identified; but the slot fall-back is driven by `bootcount`, not
  by a bad device tree, so a `reserved-memory` map that stops the kernel reaching userspace is not
  something the A/B logic is guaranteed to recover from.
- The board's only console is `ttyS0` (serial); a bad `reserved-memory` map can panic the kernel or
  fail to boot, and recovery would need the serial console or a vendor reflash.

**So the honest statement is: the ceiling is a device-tree ION pool size, in principle resizable, and
it is the `2 GiB ion_carveout` — not `bpu_region` — that binds.** The candidate is ready; whether a
larger `ion_carveout` actually boots and lets 4B load remains **unproven**, because proving it needs a
reboot this board cannot safely undo.

### Preparing the resize offline

Everything that does **not** need a boot has been done, so the eventual boot is a one-shot with a
pre-validated artifact. **No file under `/boot` was written and the live device tree was not touched**;
the work is on copies under `/tmp/s600_carveout_prep/` on the board:

| File | What |
|---|---|
| `rdk-s600-mcb-v1p0.ORIGINAL.dtb` | the live boot DTB, copied from `/boot/hobot/rdk-s600-mcb-v1p0.dtb` (md5 `35ae26dac46500abe140d1d59b669ce8`) |
| `rdk-s600-mcb-v1p0.carveout4g.dtb` | the candidate (same byte size as the original) |
| `rdk-s600-mcb-v1p0.carveout4g.dts` | candidate source |
| `reserved-memory-map.txt` / `-ORIGINAL.txt` | the `[address, size)` map of each tree |
| `roundtrip.diff` | `dtc`(candidate) − `dtc`(original) |

**The change.** Three `reg` lines move, and only three — `ion_carveout` doubles, and the two heaps
above it shift up to clear it. The four ION heaps stay a single contiguous run, so nothing below
`ion_carveout` and nothing outside the run moves:

| Node | Was `[address, size)` | Becomes |
|---|---|---|
| `ion_carveout@4140000000` | `[0x4140000000, 2.00 GiB)` | `[0x4140000000, **4.00 GiB**)` — `reg = <0x41 0x40000000 0x01 0x00000000>` |
| `ion_cma@41C0000000` | `[0x41C0000000, 1.00 GiB)` | `[0x4240000000, 1.00 GiB)` — shifted up 2 GiB |
| `ion_uncache@400000000` | `[0x4200000000, 2.00 GiB)` | `[0x4280000000, 2.00 GiB)` — shifted up 2 GiB |

**Overlap proof.** Every one of the 36 `reserved-memory` nodes' `[address, size)` was enumerated and
the enlarged run checked against all of them (including `ion_reserved@40C0000000`, `bpu_region`,
`ion_cma`, `ion_uncache`, the `vpu0/1/2_ddr_reserved` block and the `pcie*` regions): **no overlap,
before or after**. The new carveout `[0x4140000000, 0x4240000000)` begins exactly where
`ion_reserved` ends (`0x4140000000`) and ends exactly where the shifted `ion_cma` begins. The full
map is in the `reserved-memory-map*.txt` files named above.

**DRAM headroom.** The board's DRAM is 63.77 GiB (`dmesg`: `66871168K`), of which **55.48 GiB is
`MemTotal`** (usable) and the rest is kernel/firmware reserved. Everything above `bpu_region` lives in
the present-RAM window `[0x40a4000000, 0x4ffffeffff]` (61.44 GiB of DRAM). The ION run currently ends
at `0x4200000000` and would end at `0x4300000000`, still **52.0 GiB short** of the RAM top. The 2 GiB
the carveout gains is taken from general-purpose RAM (`MemTotal` falls ~2 GiB, to ~53.48 GiB) — the
board keeps ~53 GiB for userspace, which is ample.

**Round-trip.** `dtc -I dtb -O dts` on the candidate, diffed against the original, differs in
**exactly the three `reg` lines** above and nothing else (`roundtrip.diff`). But this proves the
artifact is well-formed and safe to *flash*, **not** that it boots.

### Where the bootloader gets the DTB

This was the open question; it is now answered. The S600 does **not** use the plain S100 `/boot`
files. The boot chain is a signed `miniboot` firmware (`SBL`/`spl`, packaged under
`/lib/firmware/rdk/miniboot/stable/debug/img_packages/`) that runs U-Boot, which:

1. runs A/B slot selection (`ab_select_cmd = btype ab_select bootslot`), reading the active slot from
   the SoC's always-on (`aon`) area — the live `hobotboot.slot_suffix=_a` in `/proc/cmdline` is the
   result;
2. runs `sysboot` against **`extlinux/extlinux.conf`** on the **current-slot boot partition**, which
   `/etc/hb-fstab` mounts at `/boot` (`/dev/block/platform/by-name/boot_cur` → `boot_a` = `/dev/sda12`).

So the file to edit **is** `/boot/hobot/rdk-s600-mcb-v1p0.dtb`. The board-specific label is
`drobot-s600-rdk-v1p0-kernel` — built from the board flags `hobotboot.socname=S600`,
`board.hwname=rdk`, `board.ver=V1P0` — whose lines are `lantin /lantinhv` and
`domufdt /hobot/rdk-s600-mcb-v1p0.dtb` (the `domu*` = "domain U", a guest OS under the `lantinhv`
seL4 hypervisor). `spl.img`'s U-Boot env carries exactly these label names
(`drobot-s600-rdk-v1p0-kernel`, `drobot-s600-rdk-v0p1/v0p2-kernel`, …), so the loader selects the
label from the same flags. **The reversion is a file swap**: back up the DTB, replace it, and to
revert put the backup back — but a wrong map can prevent the board from reaching userspace, so the
swap must be done **with the serial console (`ttyS0`) attached and a copy of the original in hand**,
because that is the only way back without a vendor reflash.

**The one remaining unknown for the eventual boot:** `extlinux.conf`'s `default` line is a *leftover*
S100 label (`drobot-s100-rdk-v0p5-kernel`), which is inconsistent with an S600 booting the
`drobot-s600-rdk-v1p0-kernel` label, and none of the `domu*`/`lantin` keywords is standard syslinux.
The `pxe_label`/`fdt_feat` board flags and the label list compiled into `spl.img` strongly imply the
firmware constructs the label rather than reading the `default` literally — but that is an **inference
from strings**, not something observed at boot. It does not change *which file* holds the DTB (both
the S600 label and this page name `/hobot/rdk-s600-mcb-v1p0.dtb`), but a future session should
confirm at the U-Boot prompt which label is selected before flashing anything.

### The procedure, for when a recoverable board is available

**Not run here — recorded so the eventual boot is a one-shot.** With the serial console open:

```bash
# 1. back up the live DTB (this is the only revert path)
cp /boot/hobot/rdk-s600-mcb-v1p0.dtb /boot/hobot/rdk-s600-mcb-v1p0.dtb.bak
# 2. install the pre-validated candidate from /tmp/s600_carveout_prep/
cp /tmp/s600_carveout_prep/rdk-s600-mcb-v1p0.carveout4g.dtb /boot/hobot/rdk-s600-mcb-v1p0.dtb
sync
# 3. reboot, watching ttyS0.  Verify the kernel came up with the new map:
dmesg | grep -i "Memory:"          # expect ~2 GiB less general RAM
# 4. load the shipped 4B .hbm; success is hbDNNInitializeFromFiles returning 0
#    instead of HBRT4_STATUS_RESOURCE_EXHAUSTED.
# REVERT, if it does not boot:
cp /boot/hobot/rdk-s600-mcb-v1p0.dtb.bak /boot/hobot/rdk-s600-mcb-v1p0.dtb && sync && reboot
```

For 8B (5.31 GiB), grow `ion_carveout` to 6 GiB instead and shift the same two heaps up 4 GiB; the
same overlap method applies, and the run would end at `0x4700000000` — still within present RAM, but
leaving under 48 GiB of general RAM, so re-run the map check before using it.

**The other path — and the one we can do entirely ourselves — is a `.hbm` recompiled with a smaller
footprint** — a shorter context (`cache_1024` instead of `4096`, as the SDK's VLM 7B graph uses), which
shrinks `ion_reserved` and, at a smaller `chunk`, the graph. The
compiler chain that produces one (`HF safetensors → leap_llm`/`oellm_build` → `hbdk4` → `.hbm`) runs on
**x86-64 / cp310 only**, so it is not a board-side knob. Until such a graph exists, 1.7B is the ceiling.
**That chain is scoped in [the S600 native compile chain](../architecture/s600_native_chain.md)** — it
takes the HF checkpoint directly (no GGUF leg), the compiler wheels are already in the SDK we hold
rather than behind a vendor login, and a smaller-cache 4B would have to be compiled by us on x86-64.

## Determinism

Reproducibility on this path comes entirely from the **tokenizer directory's `generation_config.json`**,
not from a flag: the delegate builds its sampler from that file when the model is loaded.

- **Greedy** (`temperature 0`, `do_sample false`) is byte-identical across runs. Measured over three
  CLI runs each: 0.6B sha256 `a154da66b9307a50`, 1.7B sha256 `bc75540e622e9551`.
- **The SDK's default is not deterministic.** The shipped `generation_config.json` uses
  `temperature 0.6`, `top_k 20`; same-prompt runs differ in the reasoning block and in surface form.
  A page that claimed "runs are reproducible" without naming the file would be wrong.

So a deterministic deployment ships a greedy `generation_config.json` beside the tokenizer. That is a
file the caller edits, which is why `--seed` is refused rather than honoured — see below.

## Accuracy against the 2080ti oracle

The project's accuracy target is agreement with the 2080ti C-engine oracle. This path is **text-in /
text-out and takes no token-id input**, so the comparison is made by feeding the oracle the exact ids
the delegate's chat template produced and diffing the greedy continuations.

**A single prompt flattered this path; a batch does not.** The first measurement used one prompt
("The capital of France is") and read as "identical for 14 tokens, then a near-tie". Widening it to
ten diverse prompts × both sizes tells a more honest story — **0/10 fully identical at either size**,
and the agreement is a **prefix**, not the whole answer:

| Size | Fully identical | Shared exact prefix (tokens) | Median |
|---|---|---|---|
| 0.6B (w8 vs oracle `q4_k_m`) | **0 / 10** | 8 … 59 | ~26 |
| 1.7B (w4 vs oracle `Qwen3-1.7B-Q4_K_M`) | **0 / 10** | 4 … 52 | ~12 |

The batch (10 prompts, their prompt ids, greedy continuation ids and text, both sizes, plus the
tokenizer and the greedy `generation_config.json`) is `~/scratch/ref/s600_parity_batch.tgz` on the
board; the single-prompt artifacts are `~/scratch/ref/parity/s600_parity.json`.

### The width confound, and why only one pair can remove it

The two rows above do **not** confound width equally, and the weaker row is the 0.6B one:

| Pair | Board `.hbm` | Oracle GGUF | Width | Scheme |
|---|---|---|---|---|
| 0.6B | `w8` (int8) | `q4_k_m` | **mismatched** (8 vs 4) | mismatched |
| 1.7B | `w4` (int4) | `Qwen3-1.7B-Q4_K_M` | matched (4 vs 4) | mismatched |

So the 0.6B divergence could be **width or scheme**, while the 1.7B row — where the 14.06-logit
Row 05 disagreement lives — isolates **scheme**, because both sides are 4-bit. A same-width 0.6B
pair would settle it, but **the vendor does not publish one**, and neither does it publish the
mirror-image lever (a `w8` 1.7B):

- **The manifest lists exactly one language `.hbm` per LLM size**, and the widths are not
  interchangeable — `resolve_model_nash-p.md` (in the SDK at
  `oellm_runtime/model/`) gives `Qwen3-0.6B → w8` and `Qwen3-1.7B → w4`, with no second row for
  either. Only the 4B lists both `w4` and `w8`.
- **The published `md5sum.txt` agrees**, listing exactly one `.hbm` under each of
  `Qwen3-0.6B/` and `Qwen3-1.7B/`.
- **The bucket itself agrees.** Probing
  `.../llm_s600/{1.0.0,1.0.2}/models/Qwen3-{0.6B,1.7B}/{w4,w8}/` across every plausible
  `chunk`/`cache` name, exactly two language files exist — the ones already on the board — and
  every other candidate returns **404** while the two shipped paths return **200**. There is no
  hidden `w4` 0.6B and no hidden `w8` 1.7B to fetch, so **no same-width pair can be built on the
  board**, and the width confound on the 0.6B row stands.

That is a finding, not a gap awaiting a retry: the removal of the confound is **vendor-gated**, and
the only path that would produce a `w4` 0.6B or a `w8` 1.7B `.hbm` is the same
[compile chain](../architecture/s600_native_chain.md) that a smaller 4B would need — built by us,
on x86-64, not fetched. Until then the honest reading is: **the 0.6B row mixes width and scheme; the
1.7B row isolates scheme; and the one large-margin divergence we have measured (Row 05) is on the
row that isolates it.**

**Most divergences are near-ties — but not all.** Row 01 (1.7B) splits on a **1.33**-logit gap, the
"two coherent phrasings of the same fact" case the single prompt showed. **Row 05 (1.7B) does not**:
the two engines diverge at the same 12-token context with the oracle scoring its own choice
**57.33** against the S600's choice **43.26** — a **14.06** gap. A gap that large is not 4-bit
rounding noise against the *same* weights; the S600's `.hbm` is therefore **not the oracle's
`q4_k_m` numerically**, whatever the bucket's `w4` label says.

### What quantization the `.hbm` actually is

The `w4`/`w8` in the filename is the **`leap_llm` weight-bit count**, and the scheme is in the
toolchain we hold (the `oellm_build` wheels shipped in the SDK — see
[the native compile chain](../architecture/s600_native_chain.md)):

- **1.7B = `w4` → symmetric, per-output-channel, weight-only affine int4.** `FakeQuantLinear`'s
  4-bit path (`leap_llm/nn/modules/linear.py:66-78`) is `q_weight = clip(round(w / scales), -7, 7)`
  with **one fp scale per output channel**, `zeros = 0` — i.e. GPTQ-**style** per-channel
  symmetric int4, but with *no* GPTQ error compensation and *no* AWQ search (neither appears in the
  shipped `llm_compression` tree).
- **0.6B = `w8` → symmetric per-channel int8** via the `const_fake_quant(..., axis=0)` path.

**This is a different scheme from `q4_k_m`, on both axes that matter.** `q4_k_m` is asymmetric
6-bit sub-scales over 256-weight *super-blocks* (a mixed `q4_K`/`q6_K` file, so `attn_v`/`ffn_down`
are 6-bit); the `.hbm` is one scale per output *row*, uniformly 4-bit (or 8-bit). There is no
`q4_k`/`q6_k`/super-block code path anywhere in `leap_llm` or `hbdk4` — the scheme is not reachable
from `leap_llm` at all. So the board `.hbm` and the oracle's GGUF are **two different quantizers**
that happen to both be "4-bit", and a 14-logit disagreement is the expected consequence, not a bug
in either engine.

**The honest statement is therefore:** the output is **coherent and on-topic** (all ten prompts
produced correct, well-formed answers — see the table below), but it is **token-identical to the
oracle only for a prefix** (median ~12 tokens at 1.7B, ~26 at 0.6B), and at least one divergence
(Row 05) is a large-margin disagreement that reflects the two sides being **different
quantizations**, not a near-tie. This path cannot be compared to 1e-3 at the logit level — the
delegate exposes no logits — so the token agreement above is the whole of the evidence, and the page
says so rather than implying more.

### The per-prompt batch (S600 side)

| Prompt | 0.6B ids | 1.7B ids | Note |
|---|---|---|---|
| factual short-answer (largest planet) | 126 | 125 | both correct |
| multi-step arithmetic (train distance) | 919 | 651 | both correct |
| code snippet (iterative factorial) | 782 | 1330 | both correct |
| list three primes > 20 | 429 | 509 | both correct |
| two-part (boiling + freezing point) | 325 | 229 | **1.7B = Row 05, the 14.06 divergence** |
| translate EN→FR | 225 | 271 | both correct |
| summarize a paragraph | 232 | 442 | both correct |
| yes/no with reasoning (is 0 even) | 251 | 401 | both correct |
| long division (100th digit of 1/7) | 789 | **4055** | 1.7B loops in ` thinking`, never closes it |
| long prefill (~225-token passage) | 120 | 122 | both correct |

Every answer is correct to a human read, including the intended-to-be-hard 1/7 prompt (the 0.6B
computes `100 mod 6 = 4` → the 4th digit of `142857` = **8**, right). The one quality defect is
1.7B's repetition loop on that prompt — a model artifact, kept in the record rather than dropped,
and a strong oracle test precisely because a degenerate loop is hard to reproduce by accident.

## Sampling and caps the CLI enforces

The delegate's sampler is **fixed when the `.hbm` is loaded**, so the sampling flags cannot be applied
on this path. The CLI's rule ([#571](https://github.com/lvyufeng/PocketLLM/pull/571),
[#572](https://github.com/lvyufeng/PocketLLM/pull/572)) is to refuse a value that would have changed
the answer, and accept one that merely spells out the default:

| Flag | On `--device horizon` |
|---|---|
| `--temperature 0`, `--top-k 0`, `--top-p 1` | **accepted** — naming the delegate's own behaviour |
| `--temperature > 0`, `--top-k > 0`, `--top-p < 1` | **refused by flag name** |
| `--min-p` (non-default) | **refused by name** — not a delegate field at all |
| `--seed` | **refused by name** — the delegate has no RNG of its own to seed |
| `--max-tokens` (≠ default) | **reported, not applied** — the delegate decodes until its own stop condition |

The refusal is the *same shared function* `server/xlm_backend.py` uses per request
(`_refuse_unsupported_sampling`), so `run` and `serve` cannot disagree about which *sampling*
values are refused. The `--max-tokens` note goes to **stderr** and the answer stays on stdout:
truncating the text *after* the delegate produced it would be the CLI inventing a cap the model never
saw.

The delegate library writes its own runtime banner to file descriptor **1**; `quiet_delegate_stdout`
(`python/pocketllm/xlm.py`) redirects fd 1 to stderr around the load and infer calls, so
`pocketllm run … > out.txt` yields a clean answer and the banner is preserved for debugging on fd 2.
See [`tests/native/test_delegate_stdout.py`](https://github.com/lvyufeng/PocketLLM/blob/main/tests/native/test_delegate_stdout.py).

## Serving on the board

`serve --device horizon` is the second entry point over this delegate
([#571](https://github.com/lvyufeng/PocketLLM/pull/571)/[#572](https://github.com/lvyufeng/PocketLLM/pull/572)),
and it is exercised end to end on the board — measured 2026-10-09 with the **0.6B (`w8`)** `.hbm`
(`Qwen3-0.6B_language_chunk_512_cache_4096_w8_nash-p_corenum_4_4.hbm`, the same checkpoint
`tests/serving/test_xlm_serve_horizon.py` hardcodes) and a greedy tokenizer directory:

- **A single request is clean.** `POST /v1/chat/completions` returns `200`,
  `Content-Type: application/json; charset=utf-8`, a body any JSON client parses, the answer in
  `message.content` and the reasoning block split into `message.reasoning_content`. No SDK-monitor
  token appears in the body.
- **The threaded server does not leak the fd-1 banner.** 6 concurrent requests × 3 rounds, all 200,
  all valid JSON, **the same answer**, and no `[UCP]`/`[DNN]`/`BPU_MONITOR`/`mod_mgr` fragment in any
  body. The banner (`[UCP]`, `[DNN]`, `[VP]`, `[HPL]`, `BPU_MONITOR`, and the loader's `mod_mgr`/
  `XlmImpl` lines) is present on the server's **stderr** and **absent from its stdout** — routed,
  not dropped, which is the property `quiet_delegate_stdout` promises under `run` and this page now
  confirms under `serve`.
- **Requests serialize.** The six concurrent requests completed in steps of ~1.37 s (a 0.6B decode:
  1.37 / 2.74 / 4.10 / 5.47 / 6.84 / 8.21 s), i.e. one at a time behind the adapter's lock, not an
  error and not an interleave — the answer was identical across the round, so no two requests shared
  the single delegate session. (The 1.7B `.hbm` was also exercised by hand and behaves the same way,
  at a larger ~2.67 s step; the committed test uses 0.6B because a smaller `.hbm` makes the 6×3 round
  cheap enough to run in CI's time budget.)
- **Per-request sampling is refused with a clean `400`.** `temperature: 0.9`, `top_p: 0.5`,
  `top_k: 20` and `min_p: 0.1` each return `400` naming the field; `temperature: 0`, `top_p: 1`,
  `top_k: 0` are accepted. No 500, no silently-ignored field.

One correction this measurement forced, and it is the reason the sentence above says *sampling*: the
two entry points did **not** fully agree on the sampling fields, because `top_k: 0` — the value the
CLI's own help and the adapter's refusal both instruct a caller to use for "no limit" — was accepted
by `run` but rejected by `serve` with `400 top_k must be >= 1` (a `SamplingParams` validator, not the
delegate refusing it). The validator now accepts `0` as "no limit" and refuses only a negative `top_k`;
`tests/serving/test_xlm_serve_horizon.py::test_naming_the_defaults_is_accepted` and
`tests/serving/test_protocol_requests.py::test_top_k_zero_means_no_limit_and_is_accepted` pin it.

Two further differences are **not** bugs, and are worth stating so the two entry points are not read
as interchangeable:

- **`serve` frames a chat prompt differently from `run`.** With no chat template available to the
  adapter, a `messages` request is rendered by the protocol layer's fallback to `"user: <text>"`,
  while `run --prompt` sends the bare `<text>`. The delegate then applies its own ChatML template to
  each, so the same user text yields a **different (both coherent) continuation**. This is the
  `render_fallback_prompt` path the adapter documents, not a serving defect — a bare-text comparison
  between `run` and `serve` is comparing two different prompts.
- **`seed` is asymmetric, and only one side changes what the path claims.** `run --seed N` is refused;
  a `serve` request with `"seed": N` is **accepted and ignored** (the same class of bug the `top_k: 0`
  fix closed, and noted here rather than fixed: it spans `native_backend` and any backend that does
  sample, so it is not this adapter's alone to correct).

## What is not on this path

- **No Python backend implements the delegate.** `--device horizon` selects the `xlm` serving adapter,
  not a [`pocketllm.backends`](../architecture/backend_model.md) implementation; the underlying
  `.hbm` is a black box that takes text and returns text.
- **No token ids, no logits.** A caller that needs per-token probabilities, a custom sampler, or the
  raw token stream cannot get them from this delegate. That needs the C engine's Qwen3 path on a host
  where it runs, or the vendor compile chain.
- **No user-defined context length.** Context is `cache_4096`, baked into the `.hbm` at compile time;
  it is not a runtime flag on this path.