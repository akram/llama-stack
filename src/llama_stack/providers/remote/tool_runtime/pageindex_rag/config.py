# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from llama_stack.core.datatypes import VectorStoresConfig


class PageIndexConfig(BaseModel):
    """Configuration for PageIndex service."""

    api_url: str = Field(
        description="PageIndex API endpoint URL"
    )
    api_key: str | None = Field(
        default=None,
        description="API key for PageIndex service (if required)"
    )
    timeout_seconds: int = Field(
        default=60,
        description="Timeout for PageIndex API calls"
    )


class PageIndexRagToolRuntimeConfig(BaseModel):
    """Configuration for vectorless RAG tool runtime using PageIndex."""

    pageindex_config: PageIndexConfig = Field(
        description="PageIndex service configuration"
    )

    # Reuse vector_stores_config for prompt templates and formatting
    # (even though we don't use vector stores, the templates are useful)
    vector_stores_config: VectorStoresConfig = Field(
        default_factory=VectorStoresConfig,
        description="Configuration for prompt templates and behavior (reused from vector stores config)",
    )

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "pageindex_config": {
                "api_url": "https://api.pageindex.example.com",
                "api_key": "${env.PAGEINDEX_API_KEY:=}",
            }
        }
