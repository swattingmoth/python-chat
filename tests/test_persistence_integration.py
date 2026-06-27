"""Integration tests for persistence against a live local Supabase stack.

These tests are opt-in because they require a running Supabase instance,
configured RPC functions, and authenticated user context.
"""

from __future__ import annotations

import os

import pytest

from python_chat.persistence import SupabaseClient

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_SUPABASE_INTEGRATION_TESTS") != "1",
    reason="Set RUN_SUPABASE_INTEGRATION_TESTS=1 with local Supabase running.",
)


def test_supabase_client_initializes_when_env_available() -> None:
    client = SupabaseClient()
    assert isinstance(client.enabled, bool)
