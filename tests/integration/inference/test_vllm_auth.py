# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
from pathlib import Path

import pytest
import yaml

from llama_stack.core.datatypes import StackRunConfig
from llama_stack.core.stack import Stack


async def test_vllm_auth_error_async():
    """Test that vLLM provider correctly handles missing authentication token."""
    config_path = Path("/Users/akram/go/src/github.com/llamastack/llama-stack/stack_config.yaml")
    config_dict = yaml.safe_load(config_path.read_text())
    run_config = StackRunConfig(**config_dict)

    with pytest.raises(ValueError) as exc_info:
        stack = Stack(run_config)
        await stack.initialize()

    assert "API key is not set" in str(exc_info.value)
    assert "vllm_api_token" in str(exc_info.value)


def test_vllm_auth_error():
    """Non-async wrapper for the async test."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(test_vllm_auth_error_async())
    finally:
        loop.close()
