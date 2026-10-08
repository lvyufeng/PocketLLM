# The board fleet

Three machines run a resident Claude Code session for this project, each in a `tmux` session named
`cc`, each driving the same model through the DeepSeek gateway `~/claude_ds.sh` configures:

| Host | Board | Role | Repository |
|---|---|---|---|
| `2080ti` | x86-64, 4× RTX 2080 Ti | x86 build host, CUDA device, CPU control | `/mnt/data1/PocketLLM` |
| `orangepi-20t` | Orange Pi AIpro 20T, Ascend 310B1 | CANN / aclnn investigation, custom kernels on the NPU | `~/PocketLLM` |
| `s600` | RDK S600, Horizon "Nash" BPU | the `xlm` / `.hbm` delegate, AOT compile work | `~/PocketLLM` |

They are reached over SSH **from one control host** (the developer's Mac). No board can SSH to
another, and none of the three can push to GitHub — the tokens there are expired and the one SSH key
is not authorized. Coordination therefore runs through the control host, not between the boards.

## The one rule

**Only one machine owns the tree. Board sessions read and investigate; they do not edit `src/`.**

Concretely, the control host owns `src/`, the branch, and the pull request. A board session may:

- read and `grep` the checkout on its own board, including a few hundred megabytes of CANN headers or
  SDK docs that would be expensive to move across the link;
- run experiments, probes and builds **on the board**, and record what it found;
- write scratch programs under `~/scratch/` on the board.

A board session must **not** commit to `src/` on its own board. The failure mode is silent: three
boards each build a shared object that runs, so nothing reports a conflict, and the divergence only
surfaces when the branches are compared — by which point three "working" engines exist and none is
the reviewed one.

This is not a claim that a board-resident session is worse at the work. It is that the *evidence*
for a change to `src/` — the diff, the tests, the review — lives in one place, and splitting the
writers splits it. Board-resident sessions earn their keep on the other axis: the SDK on the board is
enormous and only worth reading in place.

## Getting work back

Nothing on a board can push, so a change made there travels as a **git bundle**:

```bash
# on the board: the working tree is a checkout of the board's own clone
ssh <board> 'cd ~/PocketLLM && git bundle create /tmp/work.bundle <branch>'

# on the control host
scp <board>:/tmp/work.bundle /tmp/work.bundle
git bundle verify /tmp/work.bundle
git fetch /tmp/work.bundle <branch>:<branch>
```

`git bundle verify` is not optional. It is how a truncated `scp` — the failure this path is most
exposed to — is caught before an incomplete history is fetched into the tree.

## Why tmux, and not a screen window

The sessions are `tmux` so a board session can be **detached** and left running while a long build or
a model conversion proceeds, and reattached from the control host with `ssh -t <board> tmux attach -t
cc`. A session that dies with its SSH connection cannot hold a multi-hour job, which is the whole
reason to have one on the board rather than a subagent driven from the control host.

A subagent, by contrast, is right for work whose evidence is already on the control host — a build,
a test run, a change to `src/` — because it inherits the one tree and the one context instead of
answering over a link. The two are not competing: the subagent is the control host's own hands, and
the board session is a way to look at something too large to bring home.

## Setup, and reproducing it

On a board, the pieces are: the native Claude Code install (`curl -fsSL https://claude.ai/install.sh |
bash`, no Node needed), `~/claude_ds.sh` for the gateway, and a `tmux new-session -d -s cc
"~/start_cc.sh"`. `~/start_cc.sh` puts `~/.local/bin` on `PATH`, sources `claude_ds.sh`, and `cd`s to
the checkout. The trust prompt that a first headless start raises is seeded by setting
`projects["<repo>"].hasTrustDialogAccepted = true` and `hasCompletedOnboarding = true` in
`~/.claude.json`, which is more reliable than driving the dialog with synthetic keystrokes.

One note worth carrying: the tmux on a board must come from the board. A binary copied from the other
aarch64 board can fail on a glibc version older than the one it was built against — so the Orange Pi
runs a **statically linked** tmux 3.4, built on the S600, which is immune to that. Check `tmux -V`
works on the target before relying on the session.