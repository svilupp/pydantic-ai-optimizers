"""Test that imports work correctly."""


def test_main_imports():
    """Test that main package imports work."""
    from pydantic_ai_optimizers import Dataset, Optimizer, ReportCase, get_optimizer_config

    assert Optimizer is not None
    assert get_optimizer_config is not None
    assert Dataset is not None
    assert ReportCase is not None


def test_version_import():
    """Test that version can be imported."""
    import pydantic_ai_optimizers

    assert hasattr(pydantic_ai_optimizers, "__version__")
    assert pydantic_ai_optimizers.__version__ == "0.0.1"
