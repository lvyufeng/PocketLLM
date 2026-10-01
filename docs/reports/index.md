# Reports

Long-form rendered documents. This section is empty, and deliberately so rather than by oversight.

A report here is a measurement taken against a runtime, with the configuration it was taken under.
This tree is pre-release: **no device backend implements a kernel**, so there is nothing to measure
and nothing that could be reported. A performance page written today would be a claim about code that
does not exist.

The previous tree's reports — the 2080 Ti DeepSeek-V4 study among them — measured runtimes that have
since been rewritten. They belong with that code, which is preserved on the `legacy` branch, and the
multi-card half of it is in [RelicLLM](https://lvyufeng.github.io/RelicLLM/).

## What a report here will need

When there is something to measure, a report in this section carries the conditions the numbers were
taken under, not only the numbers: the checkpoint and its format, the device and the driver or
toolkit version, the prompt and its length, the warm state, and the measurement convention. A prefill
rate does not imply a decode rate, and no figure transfers between configurations.

The [kernel ABI](../architecture/kernel_abi_v1.md) and the
[conformance harness](../architecture/backend_model.md#conformance) are the two things a measurement
will lean on: the first says what the backend claimed to implement, and the second says whether it
agreed with the reference implementation before anyone timed it.