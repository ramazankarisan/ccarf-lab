"""Shared pytest fixtures. pytest loads this file automatically."""

import pytest

from ex1_support_agent import backend


@pytest.fixture(autouse=True)
def clean_backend():
    """Every test starts from the seed data and leaves nothing behind."""
    backend.reset()
    yield
    backend.reset()
