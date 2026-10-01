# Architecture

PocketLLM is built around one question — *does this checkpoint fit the device in front of me, and if
not, what is the lowest weight format that makes it fit* — and the design is what that question
forces. The pages here are the design record, in the order the layers depend on each other.

| Page | What it answers |
|---|---|
| [The kernel ABI](kernel_abi_v1.md) | What a backend must implement, what it may declare, and why the ABI is stdlib-only |
| [Backends and dispatch](backend_model.md) | How an op finds a backend, and how a device is named |
| [Execution](execution.md) | The execution layer, memory, plans, and reading a GGUF checkpoint |
| [Device targets](devices.md) | What each backend is for and what it is waiting for |

## The three words

Three similar words appear throughout the code and the documentation, and they are not
interchangeable:

- **kernel / ABI** ([`pocketllm.kernels`](kernel_abi_v1.md)) — the vocabulary. Descriptors,
  declarations, and the dispatch rule. It imports nothing.
- **backend** ([`pocketllm.backends`](backend_model.md)) — the device. It allocates on a device and
  runs ops there. Selected with `--backend`, discovered by name.
- **architecture** ([`pocketllm.architectures`](../models/architectures.md)) — the model's
  *structure*. It turns a config into a graph, names no device, and allocates nothing.

The **engine** (`pocketllm.engine`) sits between them: it takes an architecture's graph, asks the
dispatcher which backend will run each op, and drives the result on a session.

## The layering rule

The dependency directions are enforced by `tests/test_package_boundaries.py`, which parses imports
statically rather than importing modules — so a violation is reported by file and line even when the
module cannot be imported on this host. That matters more here than usual, because the whole point
is to police backends whose runtimes are absent.

```
kernels/         -> stdlib only, at any level
quant/           -> numpy (a leaf: loader and reference both need it)
loader/          -> kernels + quant + numpy
backends/        -> kernels + quant (+ its own runtime, imported lazily)
engine/          -> kernels, quant, backends, loader, architectures
architectures/   -> kernels
api/             -> api + the backend *registry* (for --device's choices)
protocol/        -> api + loader (templating reads a checkpoint's declared architecture)
server/          -> api + choices + protocol
tokenizer/       -> api + loader
cli, __init__    -> anything; the assembly point
```

Two rules do the real work:

- **The ABI imports nothing.** If `pocketllm.kernels` ever imports numpy or torch, then every backend
  needs that runtime just to read a shape.
- **A backend does not import another backend, and nothing below the engine imports a backend
  package.** A shared kernel would otherwise arrive by one backend importing another's module, and
  the two would stop being separable — which is what makes "one install, several devices" work.

A third rule is behavioural rather than structural, and is also a test: `import pocketllm` must put
neither `numpy` nor `torch` into `sys.modules`, in a fresh subprocess.