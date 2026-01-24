# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack_api import Api

from .config import PageIndexRagToolRuntimeConfig


async def get_provider_impl(config: PageIndexRagToolRuntimeConfig, deps: dict[Api, Any]):
    from .memory import PageIndexRagToolRuntimeImpl

    impl = PageIndexRagToolRuntimeImpl(
        config,
        deps[Api.inference],
        deps[Api.files],
    )
    await impl.initialize()
    return impl
