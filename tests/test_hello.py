"""Tests for the hello module.

These tests verify the placeholder functionality.
"""

from python_chat.hello import hello_world


def test_hello_world_returns_hello_world() -> None:
    """Test that hello_world returns the expected string."""
    assert hello_world() == "Hello World"


def test_hello_world_type() -> None:
    """Test that hello_world returns a string (type check via runtime too)."""
    result = hello_world()
    assert isinstance(result, str)
