# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel

from llama_stack_api import Api

from .config import PageIndexRagToolRuntimeConfig
from .memory import PageIndexRagToolRuntimeImpl


class PageIndexRagToolProviderDataValidator(BaseModel):
    pageindex_api_url: str | None = None
    pageindex_api_key: str | None = None


async def get_adapter_impl(config: PageIndexRagToolRuntimeConfig, deps):
    impl = PageIndexRagToolRuntimeImpl(
        config,
        deps[Api.inference],
        deps[Api.files],
    )
    await impl.initialize()
    return impl
