"""The reference session: the ABI's op call, carried out on host memory.

This is where "a Tensor is a descriptor bound to bytes" stops being an
abstraction and becomes a numpy array.  The whole file is that translation, in
both directions, plus the argument marshalling that lets one function signature
serve every op in the registry:

* an argument is a :class:`Tensor`, so it becomes a numpy array read out of the
  buffer through the schema's output descriptors -- widened from f16/bf16, or
  decoded from packed blocks when the descriptor is quantized;
* an argument is a scalar, and it is passed through as itself.

The quantized case is worth naming, because it is the one place the session adds
information the schema does not carry.  A :class:`QuantFormat` is a *shape* to
the ABI -- block geometry, nothing more -- and a kernel needs the blocks
*decoded*, which means it needs to know which decoder.  The session therefore
passes the format through as a keyword the kernel for that op accepts
(``w_blocks_fmt``, ``table_fmt``), named after the argument, so the two are
never out of step.

Nothing here allocates device memory or records a graph: the reference backend
reports no graph capability, and both graph methods answer ``None``, which the
engine reads as "run this eagerly" rather than as an error.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from pocketllm.kernels.backend import Backend, CompileSpec, GraphCapability
from pocketllm.kernels.buffer import Buffer, DeviceBuffer
from pocketllm.kernels.device import Device
from pocketllm.kernels.dtypes import DType
from pocketllm.kernels.errors import ShapeError
from pocketllm.kernels.registry import OPS
from pocketllm.kernels.schema import Kind
from pocketllm.kernels.tensor import Tensor, TensorDesc

from ..base import RuntimeProbe
from .dtypes import ITEMSIZE, NP_OF, read, write
from .kernels import KERNELS

__all__ = ["ReferenceSession", "ReferenceBackend"]


class ReferenceBackend:
    """The correctness oracle: numpy on the host, every op, no runtime needed.

    It is always *available* -- numpy is a base dependency -- and it is never a
    *preference*; :class:`~pocketllm.kernels.dispatch.Dispatcher` sorts it last
    among equal candidates, and a session policy can disable it entirely.  Both
    of those come from ``is_reference`` being true.
    """

    name = "reference"
    device_kind = "cpu"
    version = "0.0.0"
    summary = "Correctness oracle: numpy, host memory, every op"
    missing_dependency = "numpy"
    is_reference = True

    #: The one probe every backend has, so a caller can read ``.probe`` without
    #: asking which kind of backend it holds.  This is also what makes "always
    #: available" a *fact* rather than an assertion: numpy is a base dependency
    #: and the oracle is built on it, but the claim is now the same check the
    #: other backends run, not a constant that could go stale.
    probe = RuntimeProbe(modules=("numpy",))

    def available(self) -> bool:
        return bool(self.probe())

    def capabilities(self):
        from pocketllm.kernels.backend import Capability
        from pocketllm.kernels.dtypes import QUANT_FORMATS

        all_dtypes = frozenset({DType.F32, DType.F16, DType.BF16})
        # The reference backend reuses the *schema's* declared domains rather
        # than narrowing them: it implements every op over everything the op
        # admits, which is what "normative" means.  A capability that quietly
        # covered less would make dispatch fall to "no backend" for a call the
        # oracle could have answered.
        out = []
        for schema in OPS.schemas():
            quant_names = {fmt.name for fmt in schema.quants}
            quants = frozenset(QUANT_FORMATS[n] for n in quant_names if n in QUANT_FORMATS)
            out.append(Capability(op=schema.name, dtypes=all_dtypes, quants=quants, rank=10_000))
        return tuple(out)

    def graph(self) -> GraphCapability:
        return GraphCapability()

    def compile_spec(self) -> CompileSpec | None:
        return None

    def open(self, device: Device, *, options: Mapping[str, Any] | None = None) -> "ReferenceSession":
        return ReferenceSession(self, device)


class ReferenceSession:
    """One host memory arena, and the op implementations that read it."""

    def __init__(self, backend: ReferenceBackend, device: Device) -> None:
        self.backend = backend
        self.device = device
        self._live: list[DeviceBuffer] = []
        self._closed = False

    # -- memory -------------------------------------------------------------

    def alloc(self, nbytes: int, *, align: int = 64) -> Buffer:
        """Allocate ``nbytes``, genuinely aligned.

        numpy does not promise an alignment, so the allocation is over-sized and
        sliced at the first address that meets ``align``.  The bytes are zeroed:
        an uninitialised buffer that a kernel only partly writes is the classic
        source of a result that depends on the previous run, and a reference
        backend that did that would make the conformance harness flaky.
        """
        nbytes = int(nbytes)
        if nbytes < 0:
            raise ValueError(f"allocation size must be non-negative, got {nbytes}")
        align = max(1, int(align))
        raw = np.zeros(nbytes + align, dtype=np.uint8)
        offset = (-raw.ctypes.data) % align
        view = raw[offset : offset + nbytes]
        buffer = DeviceBuffer(
            device=self.device,
            nbytes=nbytes,
            alignment=align,
            _address=int(view.ctypes.data),
            owner=raw,
            _view=memoryview(view),
        )
        self._live.append(buffer)
        return buffer

    def free(self, buffer: Buffer) -> None:
        """Drop the session's reference to a buffer.

        numpy owns the storage, so the bytes go when the last view does; what
        this must do is stop the session from pinning them.  Freeing a buffer
        this session did not allocate is a no-op rather than an error -- a
        ``HostBuffer`` wrapping a caller's array is the common case.
        """
        try:
            self._live.remove(buffer)  # type: ignore[arg-type]
        except ValueError:
            return

    def to_device(self, host: memoryview, desc) -> Tensor:
        """Copy host bytes onto the device and describe them.

        A copy, not a view, even though the device *is* the host: the contract is
        that a session owns what it is given, and a view would make the caller's
        later write visible to a kernel that already ran.  The reference
        implementation gets this right so the fast ones cannot quietly not.
        """
        desc = _as_desc(desc)
        buffer = self.alloc(desc.nbytes, align=1)
        view = buffer.host_view()
        assert view is not None
        view[:] = bytes(host)[: desc.nbytes]
        return Tensor(desc, buffer)

    def to_host(self, tensor: Tensor) -> memoryview:
        view = tensor.buffer.host_view()
        if view is None:
            raise NotImplementedError(
                f"{self.device} cannot map its memory to the host; the reference backend always can, "
                "so this is a buffer from another session"
            )
        return view

    # -- helpers ------------------------------------------------------------

    def tensor(self, array: np.ndarray, *, dtype: DType | None = None) -> Tensor:
        """Wrap a numpy array as an ABI tensor without copying, when it can.

        The dtype is inferred from the array when not given; a c-contiguous
        array that already matches is handed over by reference, which is what
        lets a test drive a kernel with a literal and a real caller with a
        loaded weight without two code paths.

        A 0-d array -- what a scalar argument is -- is left alone rather than
        passed through ``ascontiguousarray``, which promotes it to shape ``(1,)``
        and would turn a ``shape=()`` argument into a shape error.
        """
        array = np.asarray(array)
        dtype = dtype or _dtype_of(array)
        desc = TensorDesc(tuple(array.shape), dtype=dtype)
        data = array if array.flags["C_CONTIGUOUS"] else np.ascontiguousarray(array)
        raw = data.view(np.uint8).reshape(-1)
        buffer = DeviceBuffer(
            device=self.device,
            nbytes=desc.nbytes,
            alignment=1,
            _address=int(data.ctypes.data),
            owner=data,
            _view=memoryview(raw)[: desc.nbytes],
        )
        return Tensor(desc, buffer)

    def array(self, tensor: Tensor) -> np.ndarray:
        """Read a tensor's values as a numpy array.

        A quantized tensor is *not* decoded here: the ABI gives no way to name a
        decoder, and guessing one would be worse than refusing.  Ops that take
        packed weights get their blocks raw and decode them through the format
        the session passed alongside; see :meth:`_call`.
        """
        view = tensor.buffer.host_view()
        if view is None:
            raise NotImplementedError(f"{self.device} memory is not host-mappable")
        raw = np.frombuffer(view, dtype=np.uint8, count=tensor.nbytes)
        if tensor.desc.is_quantized:
            assert tensor.quant is not None
            blocks_per_row = math.ceil(tensor.desc.packed_row_elems / tensor.quant.block_elems)
            return raw.reshape(tensor.desc.rows, blocks_per_row, tensor.quant.block_bytes)
        return read(raw, tensor.desc.dtype, tensor.desc.shape)

    # -- the op-level ABI ---------------------------------------------------

    def run(
        self,
        op: str,
        args: Sequence[Any],
        *,
        out: Sequence[Tensor] | None = None,
        attrs: Mapping[str, Any] | None = None,
    ) -> tuple[Tensor, ...]:
        """Run one op through the schema's shape inference and its kernel.

        The schema is consulted even when the kernel does not need it, because
        the shape rule is the *declaration* of what this call produces and the
        reference backend is the thing that must honour it.  A call whose args
        do not match the declared shapes fails here, before any arithmetic.
        """
        if self._closed:
            raise RuntimeError("this reference session is closed")
        schema = OPS.get(op)
        attrs = dict(attrs or {})
        descs = schema.infer(args, attrs)

        positional: list[Any] = []
        hints: dict[str, Any] = {}
        quant_fmt: dict[str, str] = {}
        for spec, value in zip(schema.args, args):
            if spec.kind is not Kind.TENSOR:
                positional.append(value)
                continue
            if value is None:
                positional.append(None)
                continue
            tensor = _as_tensor(value)
            positional.append(self.array(tensor))
            if tensor.desc.is_quantized:
                assert tensor.quant is not None
                quant_fmt[spec.name] = tensor.quant.name
        for name, fmt in quant_fmt.items():
            hints[f"{name}_fmt"] = fmt

        results = self._call(op, positional, {**attrs, **hints})
        outputs = results if isinstance(results, tuple) else (results,)

        if out is not None and len(out) != len(descs):
            raise ShapeError(f"{op}: kernel produced {len(outputs)} outputs, {len(out)} were supplied")

        tensors: list[Tensor] = []
        for index, desc in enumerate(descs):
            values = np.asarray(outputs[index])
            if out is not None:
                target = out[index]
                if target.desc.shape != desc.shape:
                    raise ShapeError(
                        f"{op}: supplied output {index} has shape {target.desc.shape}, expected {desc.shape}"
                    )
                self._store(target, values, desc.dtype)
                tensors.append(target)
            else:
                tensors.append(self._allocate_output(desc, values))
        return tuple(tensors)

    def _call(self, op: str, positional: Sequence[Any], kwargs: Mapping[str, Any]):
        try:
            kernel = KERNELS[op]
        except KeyError as exc:
            raise NotImplementedError(
                f"the reference backend has no implementation of {op!r}; "
                "a declared op is not complete until it has one"
            ) from exc
        return kernel(*positional, **kwargs)

    def _allocate_output(self, desc: TensorDesc, values: np.ndarray) -> Tensor:
        buffer = self.alloc(desc.nbytes, align=1)
        tensor = Tensor(desc, buffer)
        self._store(tensor, values, desc.dtype)
        return tensor

    def _store(self, tensor: Tensor, values: np.ndarray, dtype: DType) -> None:
        view = tensor.buffer.host_view()
        if view is None:
            raise NotImplementedError(f"{self.device} memory is not host-mappable")
        target = np.frombuffer(view, dtype=np.uint8, count=tensor.nbytes)
        write(target, values.reshape(tensor.desc.shape), dtype)

    # -- the graph path, which is deliberately absent -----------------------

    def compile_graph(self, graph):
        """No AOT path: the reference backend runs eagerly, always."""
        return None

    def capture(self, region, *, warmup: int = 3):
        """No capture path.  ``None`` means "run this eagerly", not "failed"."""
        return None

    def flush(self) -> None:
        """Nothing is buffered: every op has already finished when it returns."""
        return None

    def close(self) -> None:
        self._closed = True
        self._live.clear()


def _dtype_of(array: np.ndarray) -> DType:
    """The ABI dtype for a numpy array, refusing one the ABI cannot name."""
    lookup = {np.dtype(t): d for d, t in NP_OF.items()}
    found = lookup.get(array.dtype)
    if found is None:
        raise NotImplementedError(f"the ABI has no dtype for numpy {array.dtype}")
    return found


def _as_desc(value: Any) -> TensorDesc:
    if isinstance(value, TensorDesc):
        return value
    desc = getattr(value, "desc", None)
    if isinstance(desc, TensorDesc):
        return desc
    raise ShapeError(f"expected a tensor descriptor, got {type(value).__name__}")


def _as_tensor(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value
    raise ShapeError(f"expected a tensor, got {type(value).__name__}")


#: Re-exported so ``ITEMSIZE`` has a use outside the dtype module: a caller
#: sizing an allocation from a descriptor can ask either way.
__all__ += ["ITEMSIZE"]