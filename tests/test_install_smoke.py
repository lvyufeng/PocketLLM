"""Smoke test to verify basic import and API availability after installation."""

import sys


def test_import_pocketllm():
    """Test that pocketllm can be imported."""
    import pocketllm
    assert pocketllm.__version__ is not None


def test_public_api_available():
    """Test that public API classes are accessible."""
    import pocketllm

    # Core API classes
    assert hasattr(pocketllm, 'LLM')
    assert hasattr(pocketllm, 'AsyncLLM')
    assert hasattr(pocketllm, 'EngineArgs')
    assert hasattr(pocketllm, 'SamplingParams')
    assert hasattr(pocketllm, 'GenerationRequest')
    assert hasattr(pocketllm, 'GenerationResult')

    # Exception classes
    assert hasattr(pocketllm, 'PocketLLMError')
    assert hasattr(pocketllm, 'BackendUnavailableError')
    assert hasattr(pocketllm, 'ConfigurationError')
    assert hasattr(pocketllm, 'UnsupportedFeatureError')


def test_backends_importable():
    """Test that backend modules can be imported."""
    import pocketllm.backends
    assert pocketllm.backends is not None


def test_cli_importable():
    """Test that CLI module can be imported."""
    import pocketllm.cli
    assert hasattr(pocketllm.cli, 'main')


def test_version_format():
    """Test that version string has expected format."""
    import pocketllm
    version = pocketllm.__version__
    assert isinstance(version, str)
    parts = version.split('.')
    assert len(parts) >= 2, f"Version '{version}' should have at least major.minor"


if __name__ == '__main__':
    # Run all test functions
    test_import_pocketllm()
    test_public_api_available()
    test_backends_importable()
    test_cli_importable()
    test_version_format()
    print("✅ All smoke tests passed")
