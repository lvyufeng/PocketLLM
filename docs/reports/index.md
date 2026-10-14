# Reports

Long-form rendered documents. This section is empty, and deliberately so rather than by oversight.

A report here is a measurement taken against a runtime, with the configuration it was taken under.
This section is empty because no measurement has been taken under conditions worth recording: the C
engine runs Qwen3-0.6B correctly, but on f16 weights with a load path that still dominates decode, and
a number taken now would describe the loader rather than the kernels. It will be worth writing when
the weights are quantized — a Q4_K GEMM is where the width ladder starts to pay, and that is the
regime this repository exists for.

## What a report here will need

When there is something to measure, a report in this section carries the conditions the numbers were
taken under, not only the numbers: the checkpoint and its format, the device and the driver or
toolkit version, the prompt and its length, the warm state, and the measurement convention. A prefill
rate does not imply a decode rate, and no figure transfers between configurations.

The [kernel ABI](../architecture/kernel_abi_v1.md) and the
[conformance harness](../architecture/backend_model.md#conformance) are the two things a measurement
will lean on: the first says what the backend claimed to implement, and the second says whether it
agreed with the reference implementation before anyone timed it.