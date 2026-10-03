"""Reading a ``.gguf`` checkpoint -- and, for tests, writing a small one.

`reader` is the normative reader of the format; `tensor_reader` and
`quantized_loader` turn its records into numpy or into packed blocks.  `writer`
is the other direction and exists for a narrower reason: the tests that check the
C engine's forward pass against the Python graph need one checkpoint both can
read on a host where no real checkpoint is installed, so they write a tiny one.
It is a test fixture that happens to live in the package, not the start of a
converter.
"""
