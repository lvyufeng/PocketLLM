# `--device` splits into a platform and a card list

**Affects:** anyone passing `--device` to `pocketllm serve`, or the `device` key to
`--backend-option`, or constructing `EngineArgs(device=...)` with a card in it.

## What changed

One name did two jobs, and the two jobs disagreed: the top level refused `--device` outright under
automatic TP supervision and passed whatever it got to the native engine as a device index, while
each of the `v41`, `mimo` and `xing4` runtimes read its own `device` as a *card* — `v41` twice over,
as the card its dense tree sits on and again as the base the expert loader's rank offset is applied
to. `mimo` ignored the top level's value entirely. It is now two flags, which is what vLLM
(`--device` / `--device-ids`) and SGLang (`--device` / `--base-gpu-id`) each do.

| | Before | Now |
|---|---|---|
| `--device cuda:2` | The card, on the routes that read it | `ConfigurationError` naming `--device-ids 2` |
| `--device 0` | The card, on the routes that read it | `ConfigurationError` naming `--device-ids 0` |
| `--device cuda` (or unset) | Accepted; meant a card on some routes | The platform. `auto` (the default) asks the build |
| `--device cpu` | Accepted; meant "no card" on `xing4` only | The platform, on every runtime |
| `--device-ids 2,3` | Did not exist | The cards, in rank order: rank *r* takes the *r*-th |
| `--backend-option device=cuda:1` | The only spelling that reached `v41`/`mimo`/`xing4` | `ConfigurationError`, naming `--device-ids` — `device` is not a declared option any more |
| `DEVICE=cuda:1` | The environment bridge's spelling of the same card | `ConfigurationError`; `POCKETLLM_DEVICE_IDS=2,3` is the new one |
| `--device 0 --tensor-parallel-size 4` | `ConfigurationError`: the pair was refused under supervision | Works: a platform has nothing to conflict with supervision |
| `CUDA_VISIBLE_DEVICES=$rank … --device 0` | The rank pattern, one process at a time | Still works unchanged, and `--device-ids $rank` is the same thing written once |

`--device-ids` takes indices into the set the process can see, so the narrowed form above keeps
meaning what it meant. On an unnarrowed host `--device-ids 2,3` names physical cards 2 and 3, which
is the spelling to move to: the launcher no longer rewrites `CUDA_VISIBLE_DEVICES` per rank, and one
command line names the whole world.

## What to do

**A card under `--device`** — move the number:

```bash
pocketllm serve --model ... --device cuda:2          # before
pocketllm serve --model ... --device-ids 2           # now
```

**A rank pattern** — the pair collapses into one flag, and `CUDA_VISIBLE_DEVICES` can go:

```bash
for rank in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$rank python -m pocketllm serve --model ... \
        --tensor-parallel-size 4 --tensor-parallel-rank $rank --device 0 &
done
# now
python -m pocketllm serve --model ... --tensor-parallel-size 4 --device-ids 0,1,2,3
```

The list must name one card per rank, and a duplicate is refused: two ranks on one card is a
collective that never completes rather than a run that is slower.

**`v41`'s expert arena** — the loader is told where the split *starts* and adds the rank itself, so
what it reads is the base of a contiguous run. `--device-ids 2,3` gives it 2, and a list with a gap
(`2,4`) is refused there rather than silently placing rank 1's experts on card 3. A caller who
really means a base names it outright with `--backend-option expert_device=cuda:1`.

**`--backend-option device=...` and `DEVICE=...`** — both spellings of the same card are gone with
the declaration behind them. A launch that used either names `--device-ids` instead (or
`POCKETLLM_DEVICE_IDS` through the environment), and the refusal says so rather than only listing
the keys that remain: the list is the answer to a typo, and this is not one — the concept moved
rather than disappearing.

## What did not change

`CUDA_VISIBLE_DEVICES` and `ASCEND_RT_VISIBLE_DEVICES` are still read, and still honoured, by every
route that places a rank on a card: with no `--device-ids` a TP rank claims card
`tensor_parallel_rank` unless the launcher narrowed visibility to one card, in which case that card
is index 0 of what the process can see. The `cpp` adapter's detection of a narrowed list is the same
code it always was, moved to `pocketllm.backends.runtime_engine.card_for_rank` so that all four
adapters ask one question once.

The native binary's own `--device N` is untouched: it is a different command line (the engine's
smoke and bench front end), it never read the card list, and the launcher pattern
`CUDA_VISIBLE_DEVICES=$rank $BIN … --tp-rank $rank --device 0` in `scripts/` still means what it
meant.
